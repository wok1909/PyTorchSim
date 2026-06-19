import math # sqrt
import sympy

from typing import List, Optional

import torch
from torch import empty_strided
from torch._inductor.ir import IRNode, TensorBox, FixedLayout
from torch._inductor.virtualized import V
from torch._inductor.select_algorithm import realize_inputs
from torch.backends.cuda import flash_sdp_enabled, mem_efficient_sdp_enabled

from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplateKernel


def _make_offset_map_with_sym(strides, sym_dim, sym_stride, offset=0):
    """Like _make_offset_map but injects a block symbol ``s`` into dimension ``sym_dim``.

    The effective index for that dimension becomes ``d{sym_dim} + sym_stride * s``.
    Use this to keep ``affine.for`` bounds static and encode the block contribution
    directly inside the ``affine.apply`` call that computes the DRAM offset.

    Args:
        strides:    per-dimension DRAM strides.
        sym_dim:    which dimension carries the block symbol.
        sym_stride: multiplier for the symbol (1 for abs-position loops like FLASH
                    ``%blk``; ``BlkS`` for block-index loops like PARTIAL ``%blk``).
        offset:     constant layout offset.

    Returns:
        MLIR affine_map string with one symbol, e.g.
        ``affine_map<(d0, d1, d2)[s] -> (d0 * 8192 + (d1 + 128 * s) * 64 + d2)>``
    """
    n = len(strides)
    terms = []
    for j, sv in enumerate(strides):
        sv = int(sv)
        if sv == 0:
            continue
        if j == sym_dim:
            inner = f"d{j} + s" if sym_stride == 1 else f"d{j} + {sym_stride} * s"
            terms.append(f"({inner})" if sv == 1 else f"({inner}) * {sv}")
        else:
            terms.append(f"d{j}" if sv == 1 else f"d{j} * {sv}")
    try:
        off = int(offset)
    except (TypeError, ValueError):
        off = 0
    if off:
        terms.append(str(off))
    dim_str = ", ".join(f"d{j}" for j in range(n))
    expr = " + ".join(terms) if terms else "0"
    return f"affine_map<({dim_str})[s] -> ({expr})>"


def _make_offset_map(strides, offset=0):
    """Generate an MLIR affine_map string for a flat DRAM base-address.

    Args:
        strides: list of integer per-dimension strides.
                 A stride of 0 means the dimension does not contribute.
        offset:  constant layout offset (e.g. from IRNode.get_layout().offset).

    Returns:
        MLIR affine_map string, e.g. ``affine_map<(d0, d1) -> (d0 * 128 + d1)>``
    """
    n = len(strides)
    terms = []
    for j, s in enumerate(strides):
        s = int(s)
        if s == 1:
            terms.append(f"d{j}")
        elif s != 0:
            terms.append(f"d{j} * {s}")
    try:
        off = int(offset)
    except (TypeError, ValueError):
        off = 0
    if off:
        terms.append(str(off))
    dim_str = ", ".join(f"d{j}" for j in range(n))
    expr = " + ".join(terms) if terms else "0"
    return f"affine_map<({dim_str}) -> ({expr})>"


def _make_offset_map_gqa(strides, head_div=1, offset=0):
    """Like _make_offset_map, but for GQA the head dimension (d0) of a KV tensor
    is divided down to the KV head index.

    The flash kernel loops over query heads (%index0 in 0..n*hq) for all DMAs.
    KV tensors only have n*h heads, with g = hq // h query heads sharing each KV
    head. So a query head index0 reads KV head ``index0 floordiv g``. When
    ``head_div`` (= g) is 1 this is identical to ``_make_offset_map``.

    Args:
        strides:  per-dimension DRAM strides; dimension 0 is the head dim.
        head_div: g = hq // h. floordiv applied to d0 when > 1.
        offset:   constant layout offset.

    Returns:
        MLIR affine_map string, e.g.
        ``affine_map<(d0, d1, d2) -> ((d0 floordiv 5) * 16384 + d1 * 128 + d2)>``
    """
    try:
        g = int(head_div)
    except (TypeError, ValueError):
        g = 1
    if g <= 1:
        return _make_offset_map(strides, offset)

    n = len(strides)
    terms = []
    for j, s in enumerate(strides):
        s = int(s)
        if s == 0:
            continue
        if j == 0:
            # Map query head index0 -> KV head index0 floordiv g.
            terms.append(f"(d0 floordiv {g})" if s == 1
                         else f"(d0 floordiv {g}) * {s}")
        elif s == 1:
            terms.append(f"d{j}")
        else:
            terms.append(f"d{j} * {s}")
    try:
        off = int(offset)
    except (TypeError, ValueError):
        off = 0
    if off:
        terms.append(str(off))
    dim_str = ", ".join(f"d{j}" for j in range(n))
    expr = " + ".join(terms) if terms else "0"
    return f"affine_map<({dim_str}) -> ({expr})>"


def flash_sdpa_args(
        query : TensorBox,
        key   : TensorBox,
        value : TensorBox) -> list:
    """
    Arg processing for flash SDPA.
    Its logic is based on:
    mm_args() which is in torch._inductor.kernel.mm_common.py (142 line).
    """

    # Materialize input buffers for the codegen backend.
    query, key, value = realize_inputs(query, key, value)

    # query : (n, hq, l, e)
    # key   : (n, h, s, e)
    # value : (n, h, s, ev)
    # out   : (n, hq, l, ev)
    # n: Batch size
    # hq: query's head counts, h: key and value's head counts.
    # l: target sequence lenght and s: source sequence length.
    # e: embeding dimension of the query and key and ev: embeding dimension of the value.
    nq, hq, l, eq  = query.get_size()
    nk, hk, sk, ek = key.get_size()
    nk, hv, sv, ev = value.get_size()

    n = V.graph.sizevars.guard_equals(nq, nk)
    n = V.graph.sizevars.guard_equals(nq, nk)

    h = V.graph.sizevars.guard_equals(hk, hv)
    s = V.graph.sizevars.guard_equals(sk, sv)
    e = V.graph.sizevars.guard_equals(eq, ek)

    # While there are no theoretical requirements for e == ev,
    # this implementation currently enforces e == ev for simplicity.
    if e != ev:
        raise NotImplementedError(
            "Flash SDPA currently requires matching head dimensions between query and value (e == ev)."
        )

    # GQA support: map each query head to its KV head by grouping (hq = g * h).
    # The KV DRAM offset maps apply (index0 floordiv g); this is correct for ANY
    # batch n, because the flattened query head index0 = batch*hq + head and
    #   index0 // g = batch*h + head // g   (since hq = g*h),
    # which is exactly the flattened KV head index. So batched GQA is supported.
    if hq != h:
        if (hq % h) != 0:
            raise NotImplementedError(f"Flash SDPA GQA requires hq % h == 0 (hq: {hq}, h: {h}).")

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [n, hq, l, ev]
    )

    return [n, hq, h, l, s, e, ev, layout, query, key, value]

def calculate_scale(query: torch.Tensor, scale: float) -> float:
    """
    Calculate the scaling factor based on the head dimension if scale is None
    Otherwise, use the provided scale.
    """
    if scale is None:
        return 1.0 / math.sqrt(query.layout.size[-1])
    else:
        return scale


FLASH_SDPA_TEMPLATE = r"""
// SDPA kernel
// b = {{ b }}
// l = {{ l }}
// s = {{ s }}
// e = {{ e }}
// tile_l = {{ tile_l }}
// tile_s = {{ tile_s }}
// tile_e = {{ tile_e }}
{{kernel.def_global_vars()}}

func.func @{{ KERNEL_NAME }}{{kernel.def_kernel(inputs=[query, key, value], outputs=[out], names_str="query, key, value, out", input_reorder=input_reorder)}} {
  // Inputs
  {{ kernel.def_sram_buffer("query", q_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("key", k_tile_desc, indent_size=2) }}
  {{ kernel.def_sram_buffer("value", v_tile_desc, indent_size=2) }}

  // Output
  {{ kernel.def_sram_buffer("out", out_tile_desc, indent_size=2) }}

  // Intermediate buffers
  // mul_buffer is a matmul operand/output -> declared in io_stype (matches q/k/v/out).
  {{ kernel.def_sram_buffer("mul", mul_tile_desc, indent_size=2, dtype=io_stype) }}
  {{ kernel.def_sram_buffer("max", max_desc, indent_size=2, dtype=acc_stype) }}
  {{ kernel.def_sram_buffer("sum", sum_desc, indent_size=2, dtype=acc_stype) }}

  // Constants (online-softmax math is in acc_stype = f32)
  %c0 = arith.constant 0.0 : {{ acc_stype }}
  %c1 = arith.constant 1.0 : {{ acc_stype }}
  %c_scale = arith.constant {{ scale }} : {{ acc_stype }}
  %c_neg_inf = arith.constant -1.0e+30 : {{ acc_stype }}

  %v0_c = arith.constant dense<0.0> : vector<{{ chunk_size }}x{{ acc_stype }}>
  %v0_2x = arith.constant dense<0.0> : vector<2x{{ acc_stype }}>

  %v_neg_inf_c = arith.constant dense<-1.0e+30> : vector<{{ chunk_size }}x{{ acc_stype }}>
  %v_neg_inf_2x = arith.constant dense<-1.0e+30> : vector<2x{{ acc_stype }}>

  %v_scale = vector.broadcast %c_scale : {{ acc_stype }} to vector<{{ tile_s }}x{{ acc_stype }}>
  %v_scale_c = vector.broadcast %c_scale : {{ acc_stype }} to vector<{{ chunk_size }}x{{ acc_stype }}>

  {{ kernel.def_local_vars(indent_size=2) }}

  affine.for %index0 = 0 to {{ b }} {
    affine.for %index3 = 0 to 1 step 1 {
      affine.for %index1 = 0 to {{ l }} step {{ tile_l }} {
        %q_dram_offset = affine.apply {{ q_offset_map }}(%index0, %index1, %index3)
        {{ kernel.def_dma_op("MVIN", "query", [], q_tile_desc, indent_size=8, dram_stride=q_dram_stride, dram_offset="q_dram_offset") }}

        {{ kernel.emit_chunked_zero_init("%out_buffer", kernel.get_spad_size_per_lane(tile_l, tile_e), out_tile_desc.get_mlir_shape(io_stype), io_stype, index_prefix="0, 0", indent_size=8) }}
        affine.vector_store %v_neg_inf_2x, %max_buffer[0, 0] : {{ max_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>
        affine.vector_store %v0_2x, %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>

        %qt_buffer2D = memref.reinterpret_cast %q_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_l }}], strides: [{{ tile_l }}, 1] : {{ q_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x{{ tile_l }}x{{ io_stype }}, 1>
        %ot_buffer2D = memref.reinterpret_cast %out_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_l }}], strides: [{{ tile_l }}, 1] : {{ out_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x{{ tile_l }}x{{ io_stype }}, 1>

        affine.for %index2 = 0 to {{ s }} step {{ tile_s }} {
          %k_dram_offset = affine.apply {{ k_offset_map }}(%index0, %index2, %index3)
          {{ kernel.def_dma_op("MVIN", "key", [], k_tile_desc, indent_size=10, dram_stride=k_dram_stride, dram_offset="k_dram_offset") }}
          %v_dram_offset = affine.apply {{ v_offset_map }}(%index0, %index2, %index3)
          {{ kernel.def_dma_op("MVIN", "value", [], v_tile_desc, indent_size=10, dram_stride=v_dram_stride, dram_offset="v_dram_offset") }}

          {{ kernel.emit_chunked_zero_init("%mul_buffer", kernel.get_spad_size_per_lane(tile_s, tile_l), mul_tile_desc.get_mlir_shape(io_stype), io_stype, index_prefix="0", indent_size=10) }}

          %k_buffer2D = memref.reinterpret_cast %k_buffer to offset: [0], sizes: [{{ tile_s }}, {{ tile_e }}], strides: [{{ tile_e }}, 1] : {{ k_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_s }}x{{ tile_e }}x{{ io_stype }}, 1>
          %vt_buffer2D = memref.reinterpret_cast %v_buffer to offset: [0], sizes: [{{ tile_e }}, {{ tile_s }}], strides: [{{ tile_s }}, 1] : {{ v_tile_desc.get_mlir_shape(io_stype) }} to memref<{{ tile_e }}x{{ tile_s }}x{{ io_stype }}, 1>


          // key @ query.t (f16-uniform matmul; SA accumulates f32 internally).
          linalg.matmul
            { idx_map = array<i32: 1, 0, -1> }
            ins(%k_buffer2D, %qt_buffer2D : memref<{{ tile_s }}x{{ tile_e }}x{{ io_stype }}, 1>, memref<{{ tile_e }}x{{ tile_l }}x{{ io_stype }}, 1>)
            outs(%mul_buffer : {{ mul_tile_desc.get_mlir_shape(io_stype) }})

          // FUSED scale+max (was 2 passes): scale scores by c_scale, store the
          // scaled scores back, AND accumulate the row-max in ONE pass over tile_s.
          // f16 keeps chunk-wide extf/truncf for legal widening LMUL.
          %old_max = affine.vector_load %max_buffer[0,0] : {{ max_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>

          %chunk_max_res = affine.for %index5 = 0 to {{ tile_s }} step {{ chunk_size }} iter_args(%iter_max=%v_neg_inf_c) -> (vector<{{ chunk_size }}x{{ acc_stype }}>) {
{% if io_stype != acc_stype %}
            %sm_io  = affine.vector_load %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ chunk_size }}x{{ io_stype }}>
            %sm_f   = arith.extf %sm_io : vector<{{ chunk_size }}x{{ io_stype }}> to vector<{{ chunk_size }}x{{ acc_stype }}>
            %chunk_val = arith.mulf %sm_f, %v_scale_c : vector<{{ chunk_size }}x{{ acc_stype }}>
            %sm_out = arith.truncf %chunk_val : vector<{{ chunk_size }}x{{ acc_stype }}> to vector<{{ chunk_size }}x{{ io_stype }}>
            affine.vector_store %sm_out, %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ chunk_size }}x{{ io_stype }}>
{% else %}
            %sm_raw = affine.vector_load %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ chunk_size }}x{{ acc_stype }}>
            %chunk_val = arith.mulf %sm_raw, %v_scale_c : vector<{{ chunk_size }}x{{ acc_stype }}>
            affine.vector_store %chunk_val, %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ chunk_size }}x{{ acc_stype }}>
{% endif %}
            %local_max = arith.maximumf %chunk_val, %iter_max : vector<{{ chunk_size }}x{{ acc_stype }}>
            affine.yield %local_max : vector<{{ chunk_size }}x{{ acc_stype }}>
          } { accumulation_loop=true }

          %max_cast = vector.shape_cast %chunk_max_res : vector<{{ chunk_size }}x{{ acc_stype }}> to vector<{{ chunk_size // 2 }}x2x{{ acc_stype }}>
          %max_reduced_1 = vector.multi_reduction <maximumf>, %max_cast, %v_neg_inf_2x [0] : vector<{{ chunk_size // 2 }}x2x{{ acc_stype }}> to vector<2x{{ acc_stype }}>
          %max_shuffled = vector.shuffle %max_reduced_1, %max_reduced_1 [1, 0] : vector<2x{{ acc_stype }}>, vector<2x{{ acc_stype }}>
          %max_reduced_2 = arith.maximumf %max_reduced_1, %max_shuffled : vector<2x{{ acc_stype }}>

          %new_max = arith.maximumf %max_reduced_2, %old_max : vector<2x{{ acc_stype }}>
          affine.vector_store %new_max, %max_buffer[0, 0] : {{ max_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>


          // Compute rescale factors: exp(old_max - new_max) (acc_stype)
          %max_diff = arith.subf %old_max, %new_max : vector<2x{{ acc_stype }}>
          %max_diff_scalar = vector.extract %max_diff[0] : {{ acc_stype }} from vector<2x{{ acc_stype }}>

{% if io_stype == acc_stype %}
          %rescale_bcast_e = vector.broadcast %max_diff_scalar : {{ acc_stype }} to vector<{{ tile_e }}x{{ acc_stype }}>
          %exp_rescale_e = math.exp %rescale_bcast_e : vector<{{ tile_e }}x{{ acc_stype }}>
{% endif %}

          %rescale_bcast_2 = vector.broadcast %max_diff_scalar : {{ acc_stype }} to vector<2x{{ acc_stype }}>
          %exp_rescale_2 = math.exp %rescale_bcast_2 : vector<2x{{ acc_stype }}>
          // Scalar rescale factor extracted from the legal vector<2> exp (scalar
          // math.exp does not legalize in this pipeline).
          %exp_rescale_scalar = vector.extract %exp_rescale_2[0] : {{ acc_stype }} from vector<2x{{ acc_stype }}>


          // Rescale previous out (io_stype buffer, math in acc_stype) and sum accumulators.
          // f16 path: chunked extf/truncf (widening-convert LMUL constraint, see above).
{% if io_stype != acc_stype %}
          %exp_rescale_c = vector.broadcast %exp_rescale_scalar : {{ acc_stype }} to vector<{{ chunk_size }}x{{ acc_stype }}>
          affine.for %oindex = 0 to {{ tile_e }} step {{ chunk_size }} {
            %or_io  = affine.vector_load %ot_buffer2D[0, %oindex] : memref<{{ tile_e }}x{{ tile_l }}x{{ io_stype }}, 1>, vector<{{ chunk_size }}x{{ io_stype }}>
            %or_f   = arith.extf %or_io : vector<{{ chunk_size }}x{{ io_stype }}> to vector<{{ chunk_size }}x{{ acc_stype }}>
            %or_mul = arith.mulf %exp_rescale_c, %or_f : vector<{{ chunk_size }}x{{ acc_stype }}>
            %or_out = arith.truncf %or_mul : vector<{{ chunk_size }}x{{ acc_stype }}> to vector<{{ chunk_size }}x{{ io_stype }}>
            affine.vector_store %or_out, %ot_buffer2D[0, %oindex] : memref<{{ tile_e }}x{{ tile_l }}x{{ io_stype }}, 1>, vector<{{ chunk_size }}x{{ io_stype }}>
          }
{% else %}
          %old_out = affine.vector_load %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ acc_stype }}, 1>, vector<{{ tile_e }}x{{ acc_stype }}>
          %rescaled_out = arith.mulf %exp_rescale_e, %old_out : vector<{{ tile_e }}x{{ acc_stype }}>
          affine.vector_store %rescaled_out, %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ acc_stype }}, 1>, vector<{{ tile_e }}x{{ acc_stype }}>
{% endif %}

          %old_sum = affine.vector_load %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>
          %rescaled_sum = arith.mulf %old_sum, %exp_rescale_2 : vector<2x{{ acc_stype }}>


          // FUSED exp+sum (was 2 passes): exp(score - new_max), store probs, AND
          // accumulate the row-sum in ONE pass over tile_s. Sum is over the f32 exp
          // (pre-truncf) for better accuracy.
          %new_max_scalar = vector.extract %new_max[0] : {{ acc_stype }} from vector<2x{{ acc_stype }}>
          %new_max_bcast_c = vector.broadcast %new_max_scalar : {{ acc_stype }} to vector<{{ chunk_size }}x{{ acc_stype }}>

          %chunk_sum_res = affine.for %index5 = 0 to {{ tile_s }} step {{ chunk_size }} iter_args(%iter_sum=%v0_c) -> (vector<{{ chunk_size }}x{{ acc_stype }}>) {
{% if io_stype != acc_stype %}
            %es_io  = affine.vector_load %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ chunk_size }}x{{ io_stype }}>
            %es_f   = arith.extf %es_io : vector<{{ chunk_size }}x{{ io_stype }}> to vector<{{ chunk_size }}x{{ acc_stype }}>
{% else %}
            %es_f   = affine.vector_load %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ chunk_size }}x{{ acc_stype }}>
{% endif %}
            %es_sub = arith.subf %es_f, %new_max_bcast_c : vector<{{ chunk_size }}x{{ acc_stype }}>
            %es_exp = math.exp %es_sub : vector<{{ chunk_size }}x{{ acc_stype }}>
{% if io_stype != acc_stype %}
            %es_out = arith.truncf %es_exp : vector<{{ chunk_size }}x{{ acc_stype }}> to vector<{{ chunk_size }}x{{ io_stype }}>
            affine.vector_store %es_out, %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ chunk_size }}x{{ io_stype }}>
{% else %}
            affine.vector_store %es_exp, %mul_buffer[0, %index5] : {{ mul_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ chunk_size }}x{{ acc_stype }}>
{% endif %}
            %local_sum = arith.addf %es_exp, %iter_sum : vector<{{ chunk_size }}x{{ acc_stype }}>
            affine.yield %local_sum : vector<{{ chunk_size }}x{{ acc_stype }}>
          } { accumulation_loop=true }

          %zero_2x = vector.broadcast %c0 : {{ acc_stype }} to vector<2x{{ acc_stype }}>
          %sum_cast = vector.shape_cast %chunk_sum_res : vector<{{ chunk_size }}x{{ acc_stype }}> to vector<{{ chunk_size // 2 }}x2x{{ acc_stype }}>
          %sum_reduced_1 = vector.multi_reduction <add>, %sum_cast, %zero_2x [0] : vector<{{ chunk_size // 2 }}x2x{{ acc_stype }}> to vector<2x{{ acc_stype }}>
          %sum_shuffled = vector.shuffle %sum_reduced_1, %sum_reduced_1 [1, 0] : vector<2x{{ acc_stype }}>, vector<2x{{ acc_stype }}>
          %sum_reduced_2 = arith.addf %sum_reduced_1, %sum_shuffled : vector<2x{{ acc_stype }}>

          %new_sum = arith.addf %sum_reduced_2, %rescaled_sum :  vector<2x{{ acc_stype }}>
          affine.vector_store %new_sum, %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>


          // value.t @ mul (io_stype-uniform matmul)
          linalg.matmul
            { idx_map = array<i32: 2, 1, -1> }
            ins(%vt_buffer2D, %mul_buffer : memref<{{ tile_e }}x{{ tile_s }}x{{ io_stype }}, 1>, {{ mul_tile_desc.get_mlir_shape(io_stype) }})
            outs(%ot_buffer2D : memref<{{ tile_e }}x{{ tile_l }}x{{ io_stype }}, 1>)
        } { accumulation_loop=true }

        // out @ row_sum^(-1) (reciprocal acc_stype; out buffer io_stype)
        %final_row_sum = affine.vector_load %sum_buffer[0, 0] : {{ sum_desc.get_mlir_shape(acc_stype) }}, vector<2x{{ acc_stype }}>
        %one_2x = vector.broadcast %c1 : {{ acc_stype }} to vector<2x{{ acc_stype }}>

        %reciprocal_row_sum_2x = arith.divf %one_2x, %final_row_sum : vector<2x{{ acc_stype }}>
        %reciprocal_scalar = vector.extract %reciprocal_row_sum_2x[0] : {{ acc_stype }} from vector<2x{{ acc_stype }}>

{% if io_stype != acc_stype %}
        // f16 path: chunked extf/truncf (widening-convert LMUL constraint, see above).
        %reciprocal_bcast_c = vector.broadcast %reciprocal_scalar : {{ acc_stype }} to vector<{{ chunk_size }}x{{ acc_stype }}>
        affine.for %nindex = 0 to {{ tile_e }} step {{ chunk_size }} {
          %fn_io  = affine.vector_load %ot_buffer2D[0, %nindex] : memref<{{ tile_e }}x{{ tile_l }}x{{ io_stype }}, 1>, vector<{{ chunk_size }}x{{ io_stype }}>
          %fn_f   = arith.extf %fn_io : vector<{{ chunk_size }}x{{ io_stype }}> to vector<{{ chunk_size }}x{{ acc_stype }}>
          %fn_mul = arith.mulf %fn_f, %reciprocal_bcast_c : vector<{{ chunk_size }}x{{ acc_stype }}>
          %fn_out = arith.truncf %fn_mul : vector<{{ chunk_size }}x{{ acc_stype }}> to vector<{{ chunk_size }}x{{ io_stype }}>
          affine.vector_store %fn_out, %ot_buffer2D[0, %nindex] : memref<{{ tile_e }}x{{ tile_l }}x{{ io_stype }}, 1>, vector<{{ chunk_size }}x{{ io_stype }}>
        }
{% else %}
        %reciprocal_bcast_e = vector.broadcast %reciprocal_scalar : {{ acc_stype }} to vector<{{ tile_e }}x{{ acc_stype }}>
        %accumulated_out = affine.vector_load %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ acc_stype }}, 1>, vector<{{ tile_e }}x{{ acc_stype }}>
        %stable_final_out = arith.mulf %accumulated_out, %reciprocal_bcast_e : vector<{{ tile_e }}x{{ acc_stype }}>
        affine.vector_store %stable_final_out, %ot_buffer2D[0, 0] : memref<{{ tile_e }}x{{ tile_l }}x{{ acc_stype }}, 1>, vector<{{ tile_e }}x{{ acc_stype }}>
{% endif %}

        %out_dram_offset = affine.apply {{ out_offset_map }}(%index0, %index1, %index3)
        {{ kernel.def_dma_op("MVOUT", "out", [], out_tile_desc, indent_size=8, dram_stride=out_dram_stride, dram_offset="out_dram_offset") }}
      } { outer_loop=true }
    } { outer_loop=true }
  } { outer_loop=true }
  return
}
"""


# ---------------------------------------------------------------------------
# Causal (prefill) variant. Identical to FLASH_SDPA_TEMPLATE, but masks scores
# to a finite -1e30 where the key position is in the future of the query, BEFORE
# the online-softmax max/exp. mul_buffer holds K@Q^T with layout [key=tile_s,
# query=tile_l], so the mask condition is (index2 + r) > (index1 + c).
# NOTE: the per-element axis mapping (whether a vector<chunk_size> spans keys or
# queries within a lane) is layout-dependent and MUST be validated numerically
# against F.scaled_dot_product_attention(is_causal=True) on a small case.
# ---------------------------------------------------------------------------
_CAUSAL_MASK_BLOCK = r"""
          // ---- Causal mask: score = -1e30 where global_key > global_query ----
          // qpos_buffer[0,0] holds this lane's query position = index1 + lane_id
          // (filled once per query-tile via MVIN of a host iota along vlane_split_axis=0).
          // Per-lane vle of the full chunk_size vector (all elements = this lane query
          // position). NO vector.extract -> stays in vector regs -> compiles to vle
          // (per-lane), not a scalar flw that would read lane 0 for every lane.
          %qpos_bcast  = affine.vector_load %qpos_buffer[0, 0] : {{ qpos_desc.get_mlir_shape(acc_stype) }}, vector<{{ chunk_size }}x{{ acc_stype }}>
          affine.for %mindex = 0 to {{ tile_s }} step {{ chunk_size }} {
            // within-lane key positions: index2 + mindex + (0..chunk_size-1)
            %kbase_idx  = arith.addi %index2, %mindex : index
            %kbase_i32  = arith.index_cast %kbase_idx : index to i32
            %kbase_f    = arith.sitofp %kbase_i32 : i32 to {{ acc_stype }}
            %kbase_vec  = vector.broadcast %kbase_f : {{ acc_stype }} to vector<{{ chunk_size }}x{{ acc_stype }}>
            %kiota_idx  = vector.step : vector<{{ chunk_size }}xindex>
            %kiota_i32  = arith.index_cast %kiota_idx : vector<{{ chunk_size }}xindex> to vector<{{ chunk_size }}xi32>
            %kiota_f    = arith.sitofp %kiota_i32 : vector<{{ chunk_size }}xi32> to vector<{{ chunk_size }}x{{ acc_stype }}>
            %kpos_vec   = arith.addf %kbase_vec, %kiota_f : vector<{{ chunk_size }}x{{ acc_stype }}>
            // mask iff global_key > global_query
            %mask_pred  = arith.cmpf ogt, %kpos_vec, %qpos_bcast : vector<{{ chunk_size }}x{{ acc_stype }}>
{% if io_stype != acc_stype %}
            %cur_scores_io = affine.vector_load %mul_buffer[0, %mindex] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ chunk_size }}x{{ io_stype }}>
            %cur_scores = arith.extf %cur_scores_io : vector<{{ chunk_size }}x{{ io_stype }}> to vector<{{ chunk_size }}x{{ acc_stype }}>
            %masked     = arith.select %mask_pred, %v_neg_inf_c, %cur_scores : vector<{{ chunk_size }}xi1>, vector<{{ chunk_size }}x{{ acc_stype }}>
            %masked_io  = arith.truncf %masked : vector<{{ chunk_size }}x{{ acc_stype }}> to vector<{{ chunk_size }}x{{ io_stype }}>
            affine.vector_store %masked_io, %mul_buffer[0, %mindex] : {{ mul_tile_desc.get_mlir_shape(io_stype) }}, vector<{{ chunk_size }}x{{ io_stype }}>
{% else %}
            %cur_scores = affine.vector_load %mul_buffer[0, %mindex] : {{ mul_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ chunk_size }}x{{ acc_stype }}>
            %masked     = arith.select %mask_pred, %v_neg_inf_c, %cur_scores : vector<{{ chunk_size }}xi1>, vector<{{ chunk_size }}x{{ acc_stype }}>
            affine.vector_store %masked, %mul_buffer[0, %mindex] : {{ mul_tile_desc.get_mlir_shape(acc_stype) }}, vector<{{ chunk_size }}x{{ acc_stype }}>
{% endif %}
          }
          // ---- end causal mask ----
"""
# Inject the per-lane query-position MVIN (once per query-tile) into the causal
# template, just before the KV loop. The marker is the score-buffer init that
# opens each KV-loop body's predecessor region.
_QPOS_MVIN = r"""
        // Per-lane query position iota for this query-tile (qpos[c] = index1 + c).
        %qpos_dram_offset = affine.apply {{ qpos_offset_map }}(%index1)
        {{ kernel.def_dma_op("MVIN", "qpos", [], qpos_desc, indent_size=8, dram_stride=qpos_dram_stride, dram_offset="qpos_dram_offset") }}
"""
FLASH_SDPA_CAUSAL_TEMPLATE = (
    FLASH_SDPA_TEMPLATE
    # add qpos as the 4th kernel input (DRAM iota = query positions)
    .replace(
        'inputs=[query, key, value], outputs=[out], names_str="query, key, value, out"',
        'inputs=[query, key, value, qpos], outputs=[out], names_str="query, key, value, qpos, out"',
        1,
    )
    # declare the per-lane qpos SRAM buffer
    .replace(
        '  {{ kernel.def_sram_buffer("value", v_tile_desc, indent_size=2) }}\n',
        '  {{ kernel.def_sram_buffer("value", v_tile_desc, indent_size=2) }}\n  {{ kernel.def_sram_buffer("qpos", qpos_desc, indent_size=2, dtype=acc_stype) }}\n',
        1,
    )
    # MVIN the per-lane query iota once per query-tile, before the KV loop
    .replace(
        "        affine.for %index2 = 0 to {{ s }} step {{ tile_s }} {",
        _QPOS_MVIN + "        affine.for %index2 = 0 to {{ s }} step {{ tile_s }} {",
        1,
    )
    # Inject the causal mask right after the QK matmul and before the FUSED
    # scale+max pass (so the row-max sees the masked scores). NOTE: the previous
    # anchor "// Find new max." was removed when scale+max were fused into one
    # pass, which silently made this .replace a no-op -> the mask was never
    # emitted and causal attention computed the non-causal result. Anchor on the
    # FUSED scale+max comment, which exists in the current template.
    .replace(
        "          // FUSED scale+max (was 2 passes): scale scores by c_scale, store the",
        _CAUSAL_MASK_BLOCK + "          // FUSED scale+max (was 2 passes): scale scores by c_scale, store the",
        1,
    )
)


class MLIRFlashSDPATemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, scale, input_reorder=None, is_causal=False, is_decode=False):
        super().__init__("kernel", input_nodes, layout, input_reorder)
        self.scale = scale
        self.is_causal = is_causal
        # Decode (single query token, L==1): reinterpret the VPU lane axis to be the
        # query-head-group g = Hq/Hkv instead of the query sequence L. One (batch, kv_head)
        # group is processed per outer iteration so every lane shares the same K/V.
        self.is_decode = is_decode

    def render(self,
               kernel: MLIRTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               prologue_nodes: Optional[List[IRNode]] = None,
               tile_info = None,
               **kwargs):

        # Except for kernel, other arguments are usually None.
        query, key, value, out, q_tensor, k_tensor, v_tensor, out_tensor, b, l, s, e, ev, n_extra_node, n_prologue_node = self.extract_info(template_buffer_node, epilogue_nodes, prologue_nodes)

        # Causal mask uses a 4th input (per-lane query-position iota); None for non-causal.
        qpos = self.input_nodes[3] if self.is_causal else None

        # Decode: lane axis = query-head-group g (not L). Recompute the outer-loop count
        # b and the lane-extent l from the ORIGINAL 4D layout (NOT the flattened b=N*Hq
        # used for prefill). One (batch, kv_head) group per outer iteration; g query heads
        # map one-per-lane within a group, so every lane in a tile shares the same K/V.
        if self.is_decode:
            # Decode (L==1): treat each query head as a degenerate prefill. Broadcast
            # its single token across ALL vector_lane lanes (l = vector_lane; q/out
            # per-lane stride 0). The outer loop runs over every query head (b = N*Hq);
            # KV is mapped query-head -> kv-head by floordiv(g) exactly like prefill
            # GQA, so the g consecutive heads sharing a KV head reuse it from L2 (no
            # extra DRAM traffic). Filling all lanes is REQUIRED: the systolic matmul
            # writeback only lays out [M=tile_e within-lane, N=tile_l across-lane]
            # correctly when the query-tile loop bound (l) and tile_l == vector_lane;
            # l=1 (one lane) collapses the output to a single D element.
            n_orig  = int(query.get_layout().size[0])
            hq_orig = int(query.get_layout().size[-3])
            b = n_orig * hq_orig         # one outer iteration per query head
            l = kernel.vector_lane       # broadcast the single token across all lanes

        if tile_info is None:
            tile_l, tile_s, tile_e, subtile_l, subtile_s, subtile_e = self.select_tile(kernel, l, s, e, n_extra_node, 0, n_prologue_node)[0]
        else:
            tile_l, tile_s, tile_e, subtile_l, subtile_s, subtile_e = tile_info

        # Decode g==1 broadcasts the single query across all tile_l(=vector_lane)
        # lanes, so the VALID lane extent is tile_l, not l(=1). Using l here would
        # set loop_size[lane]=1 and the matmul would only populate one lane/column.
        if self.is_decode and l == 1:
            TOG_latency = tile_l
        else:
            TOG_latency = l if tile_l > l else tile_l
        kernel.loop_size = [TOG_latency, tile_s, tile_e]

        # Select template code
        # Other templates will be added according to situations.
        nr_reduction_nodes = [node for node in epilogue_nodes if node.is_reduction()] if epilogue_nodes is not None else []
        if nr_reduction_nodes:
            raise NotImplementedError("FLASH_SDPA_REDUCTION_TEMPLATE is not implemented yet.")
        elif prologue_nodes:
            raise NotImplementedError("FLASH_SDPA_PROLOGUE_TEMPLATE is not implemented yet.")
        elif self.is_causal:
            template = FLASH_SDPA_CAUSAL_TEMPLATE
            epilogue_dim_aliasing = {"index0":"index0", "index1":"index1", "index2": "index2", "index3": "index3"}
            nr_rdim = 0
        else:
            template = FLASH_SDPA_TEMPLATE
            epilogue_dim_aliasing = {"index0":"index0", "index1":"index1", "index2": "index2", "index3": "index3"}
            nr_rdim = 0

        # Prepare tile descriptors for input and output tensors.
        # Intermediate buffers (transient data) do not require DRAM settings(dram stride and dram indices)
        # as they are not synchronized with external DRAM.
        # DRAM and SRAM tile shapes must match.
        vlane_stride = 1

        # (n, l, s, e, ev)
        loop_dim = [sympy.Symbol("index0"), sympy.Symbol("index1"), sympy.Symbol("index2"), sympy.Symbol("index3")]


        # Hardware constraint: The tile split axis is restricted.
        # To accommodate this, we compute (key @ query.t) instead of (query @ key.t).
        # SRAM settings
        vlane_split_axis = 1
        q_tile_size = [1, tile_l, tile_e]
        q_tile_stride = [0, tile_e, 1]
        q_tile_desc = mlir_common.MLIRMultiDimTile(q_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        q_tile_desc.set_tile_size_stride(q_tile_size, q_tile_stride)
        q_tile_desc.set_name("q_buffer")
        q_tile_desc.offset = query.get_layout().offset
        # DRAM settings
        q_stride = q_tensor.stride()

        # Since we use a weight-stationary approach in the Systolic Array (SA),
        # the split axis of the first operand differs from a standard linear algebra matmul.
        # The first operand (key) must be split along the column axis.
        # This logic aligns with the relationship between the dot product's summation direction and the hardware's accumulation direction in the SA.
        # SRAM settings
        vlane_split_axis = 2
        k_tile_size = [1, tile_s, tile_e]
        k_tile_stride = [0, 1, tile_s]
        k_tile_desc = mlir_common.MLIRMultiDimTile(k_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        k_tile_desc.set_tile_size_stride(k_tile_size, k_tile_stride)
        k_tile_desc.set_name("k_buffer")
        k_tile_desc.offset = key.get_layout().offset
        # DRAM settings
        k_stride = k_tensor.stride()

        # Since we compute mul = key @ query.t, we perform out.t = (value.t @ Softmax(mul).t).t,
        # which simplifies to (value.t @ Softmax(mul))
        # SRAM settings
        vlane_split_axis = 1
        v_tile_size = [1, tile_s, tile_e]
        v_tile_stride = [0, tile_e, 1]
        v_tile_desc = mlir_common.MLIRMultiDimTile(v_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        v_tile_desc.set_tile_size_stride(v_tile_size, v_tile_stride)
        v_tile_desc.set_name("v_buffer")
        v_tile_desc.offset = value.get_layout().offset
        # DRAM settings
        v_stride = v_tensor.stride()

        # Output is also stored in transposed format to match the value.t @ Softmax(mul) operation.
        # SRAM settings
        vlane_split_axis = 1
        out_tile_size = [1, tile_l, tile_e]
        out_tile_stride=[0, tile_e, 1]
        out_tile_desc = mlir_common.MLIRMultiDimTile(out_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        out_tile_desc.set_tile_size_stride(out_tile_size, out_tile_stride)
        out_tile_desc.set_name("out_buffer")
        # DRAM settings
        out_stride = out.get_layout().stride[1:]

        # Intermediate buffers

        # For mul = key @ query.t
        vlane_split_axis = 1
        mul_tile_size = [tile_s, tile_l]
        mul_tile_stride = [tile_l, 1]
        mul_tile_desc = mlir_common.MLIRMultiDimTile(mul_tile_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        mul_tile_desc.set_tile_size_stride(mul_tile_size, mul_tile_stride)
        mul_tile_desc.set_name("mul_buffer")
        #FIXME. What is the offset? -> It doesn't matter at this time.

        # For storing maximum values per row
        vlane_split_axis = 0
        max_size = [tile_l, 2]
        max_stride = [2, 1]
        max_desc = mlir_common.MLIRMultiDimTile(max_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        max_desc.set_tile_size_stride(max_size, max_stride)
        max_desc.set_name("max_buffer")

        # For storing summation per row
        vlane_split_axis = 0
        sum_size = [tile_l, 2]
        sum_stride = [2, 1]
        sum_desc = mlir_common.MLIRMultiDimTile(sum_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        sum_desc.set_tile_size_stride(sum_size, sum_stride)
        sum_desc.set_name("sum_buffer")

        # For reduction.
        # chunk_size = the per-op vector width of the online-softmax loops. The
        # sim charges 1 cycle per vector instruction regardless of width, so wider
        # chunks => fewer loop iterations => far fewer cycles (measured: 8->32 halves).
        # Cap at 32 (f16 widening LMUL constraint), bounded by tile_s, power of 2.
        chunk_size = max(2, min(32, tile_s))
        # Decode (is_decode): CALIBRATION fitted to measured v6e splash decode
        # device-time (per-key softmax throughput ~16x lower than the wide-vector
        # default); chunk=4 lands decode within ~30 percent of real for S>=512.
        # Prefill keeps the wide chunk. Not derived from N-UPU hardware.
        if self.is_decode:
            chunk_size = 4
        import os as _os
        _cs = _os.environ.get("TORCHSIM_FLASH_CHUNK")
        if _cs:
            chunk_size = int(_cs)

        # Per-lane query-position buffer (causal mask only): each lane holds its query
        # position (index1 + lane_id), replicated chunk_size times so the mask reads it
        # as a plain vector<chunk_size> via a per-lane vle. CRITICAL: the position MUST
        # stay in the vector domain. A vector<2> load + vector.extract[0] folds to a
        # SCALAR flw (confirmed in asm), which ignores the per-lane vu_sram_byte offset
        # and reads lane 0 for every lane -- so the whole tile gets the tile-base
        # position. Loading the full chunk_size vector and comparing it directly keeps
        # it per-lane, exactly like mul_buffer. The MVIN broadcasts one DRAM iota
        # element across the chunk axis (dram_stride 0) so all copies equal the row.
        vlane_split_axis = 0
        qpos_size = [tile_l, chunk_size]
        qpos_stride = [chunk_size, 1]
        qpos_desc = mlir_common.MLIRMultiDimTile(qpos_size, kernel.vector_lane, vlane_split_axis, vlane_stride)
        qpos_desc.set_tile_size_stride(qpos_size, qpos_stride)
        qpos_desc.set_name("qpos_buffer")

        # DMA strides and offset affine maps (dram_stride + dram_offset style)
        q_dram_stride  = [int(q_stride[0]), int(q_stride[1]), int(q_stride[2])]
        k_dram_stride  = [int(k_stride[0]), int(k_stride[1]), int(k_stride[2])]
        v_dram_stride  = [int(v_stride[0]), int(v_stride[1]), int(v_stride[2])]
        out_dram_stride = [int(out_stride[0]), int(out_stride[1]), int(out_stride[2])]

        # GQA: query has hq heads (loop %index0 in 0..n*hq), KV has h heads with
        # g = hq // h query heads per KV head. Map query head -> KV head via
        # floordiv on the KV offset maps only. g == 1 reduces to plain MHA.
        hq = int(query.get_layout().size[-3])
        h  = int(key.get_layout().size[-3])
        g  = hq // h if h else 1

        if self.is_decode:
            # index0 = query head (0..N*Hq). Broadcast the single decode token across
            # all lanes: per-lane (vlane, axis 1) DRAM stride 0 for q (read one head
            # into every lane) and out (every lane writes the identical result to one
            # DRAM address -> unique-addr model keeps write traffic flat). KV head =
            # index0 floordiv g (prefill GQA mapping); g consecutive heads share it.
            q_head_stride   = int(query.get_layout().stride[-3])
            q_d_stride      = int(query.get_layout().stride[-1])
            out_head_stride = int(out.get_layout().stride[-3])
            out_d_stride    = int(out.get_layout().stride[-1])
            # [query-head base stride, per-lane stride (0 = broadcast), D stride]
            q_dram_stride   = [q_head_stride, 0, q_d_stride]
            out_dram_stride = [out_head_stride, 0, out_d_stride]
            # K/V: %index0 already indexes the flattened kv-group (batch*Hkv+kv_head), whose
            # dim-0 stride after the [-1,S,D] view is exactly S*D. So all lanes in the group
            # read the SAME KV head with a plain offset map (no GQA floordiv needed here).
            q_offset_map   = _make_offset_map(q_dram_stride,   q_tile_desc.offset)
            k_offset_map   = _make_offset_map_gqa(k_dram_stride, g, k_tile_desc.offset)
            v_offset_map   = _make_offset_map_gqa(v_dram_stride, g, v_tile_desc.offset)
            out_offset_map = _make_offset_map(out_dram_stride, 0)
        else:
            q_offset_map   = _make_offset_map(q_dram_stride,   q_tile_desc.offset)
            k_offset_map   = _make_offset_map_gqa(k_dram_stride, g, k_tile_desc.offset)
            v_offset_map   = _make_offset_map_gqa(v_dram_stride, g, v_tile_desc.offset)
            out_offset_map = _make_offset_map(out_dram_stride, 0)

        # Causal-mask iota: [l, 2] DRAM tensor (row i = [i, i]), distributed one row
        # per lane. Tile is [tile_l, 2]; base address for query-tile index1 is index1*2.
        qpos_dram_stride = [2, 0]
        qpos_offset_map  = _make_offset_map([2], 0)

        # Keep out_idx only for epilogue_info (not in render_options)
        out_idx = [loop_dim[0]*out_stride[0], loop_dim[1]*out_stride[1], loop_dim[3]*out_stride[2]]

        kernel.render_options = dict(
            KERNEL_NAME = self.name,
            kernel = kernel,
            b = b,
            l = l,
            s = s,
            e = e,                             # Input sizes (dram)
            tile_l = tile_l,
            tile_s = tile_s,
            tile_e = tile_e,                   # Tile sizes (sram)
            # Dtype scheme (Strategy B): io_stype = model/DRAM dtype (f16/bf16/f32),
            # used for q/k/v/out + mul matmul buffers; acc_stype = f32 for the
            # online-softmax math (max/sum/exp/rescale/normalize).
            io_stype = mlir_common.DTYPE_TO_MLIR[query.get_layout().dtype],
            acc_stype = "f32",
            query = query,
            key = key,
            value = value,
            qpos = qpos,
            out = out,                         # Inputs and output (dram)
            q_dram_stride  = q_dram_stride,
            k_dram_stride  = k_dram_stride,
            v_dram_stride  = v_dram_stride,
            out_dram_stride = out_dram_stride, # Per-dim DRAM strides
            q_offset_map   = q_offset_map,
            k_offset_map   = k_offset_map,
            v_offset_map   = v_offset_map,
            out_offset_map = out_offset_map,   # Affine maps for base address
            q_tile_desc = q_tile_desc,
            k_tile_desc = k_tile_desc,
            v_tile_desc = v_tile_desc,
            mul_tile_desc = mul_tile_desc,
            out_tile_desc = out_tile_desc,     # Tile descriptions (sram)
            max_desc = max_desc,
            sum_desc = sum_desc,               # Intermediate buffer descriptions (sram)
            scale = self.scale,
            chunk_size = chunk_size,
            is_causal = self.is_causal,
            qpos_desc = qpos_desc,
            qpos_dram_stride = qpos_dram_stride,
            qpos_offset_map = qpos_offset_map,
            input_reorder = self.input_reorder # ETC
        )

        code = self._template_from_string(template).render(**kernel.render_options)
        kernel.add_loop_info([kernel.render_options["l"], kernel.render_options["s"], kernel.render_options["e"]], [kernel.render_options["tile_l"], kernel.render_options["tile_s"], kernel.render_options["tile_e"]])
        return code

    def extract_info(self, template_buffer_node, epilogue_nodes, prologue_nodes):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node

        query = self.input_nodes[0]
        key = self.input_nodes[1]
        value = self.input_nodes[2]
        out = self.output_node

        q_tensor = empty_strided(query.layout.size, query.layout.stride)
        k_tensor = empty_strided(key.layout.size, key.layout.stride)
        v_tensor = empty_strided(value.layout.size, value.layout.stride)
        out_tensor = empty_strided(out.layout.size, out.layout.stride)

        # Flatten batch and head dimensions (n, h) into a single dimension (b = n*h)
        q_tensor = q_tensor.view([-1, q_tensor.shape[-2], q_tensor.shape[-1]])
        k_tensor = k_tensor.view([-1, k_tensor.shape[-2], k_tensor.shape[-1]])
        v_tensor = v_tensor.view([-1, v_tensor.shape[-2], v_tensor.shape[-1]])
        out_tensor = out_tensor.view([-1, out_tensor.shape[-2], out_tensor.shape[-1]])

        b, l, s, e, ev = q_tensor.size(0), q_tensor.size(1), k_tensor.size(1), k_tensor.size(2), v_tensor.size(2)

        n_extra_node = len(epilogue_nodes) if epilogue_nodes is not None else 0
        n_prologue_node = len(prologue_nodes) if prologue_nodes is not None else 0

        return query, key, value, out, q_tensor, k_tensor, v_tensor, out_tensor, b, l, s, e, ev, n_extra_node, n_prologue_node

    # Reuse the existing function in MLIRBMMTemplate.
    def select_tile(self, kernel, l, s, e, n_extra_node, n_extra_read, n_prologue_node):

        # FIXME: Update the method for getting tile candidates once TestDmaFineGrained oass works correctly with Flash Attention.
        # tile_candidates = kernel.flash_sdpa_mapping(l, s, e, n_extra_node=n_extra_node)
        # tile_s (KV block) is overridable for tile-size sweeps; default = vector_lane.
        import os as _os
        _ts = _os.environ.get("TORCHSIM_FLASH_TILE_S")
        _tile_s = min(int(_ts), s) if _ts else kernel.vector_lane
        # Decode (is_decode): only l = g = Hq/Hkv query heads are valid, and g is
        # typically < vector_lane. A full vector_lane-wide tile_l would (1) read
        # garbage in the unused lanes (q DRAM only has g heads) and (2) make the
        # outer l-loop bound (g) < step (vector_lane), tripping the lane-invariant
        # assumptions in the transposed reinterpret_cast and the loop-padding pass.
        # Clamp tile_l to l so the tile uses exactly the valid lanes. Prefill keeps
        # tile_l = vector_lane (its l = query-seq is always a multiple of it).
        if self.is_decode:
            # g==1 (MHA-decode, l==1): pad tile_l to vector_lane and broadcast the
            # single query across all lanes (q/out per-lane stride 0). The systolic
            # matmul writeback only lays out [M=tile_e within-lane, N=tile_l across-
            # lane] correctly when N==vector_lane; tile_l=1 scatters M across lanes
            # and the per-lane readback recovers only a strided fraction. All lanes
            # recompute the same query (lanes are parallel -> latency-neutral) and
            # MVOUT broadcasts back to one address (unique-addr DRAM model -> no
            # extra traffic). g>1 (GQA) needs per-head lane-block replication (TODO).
            _tile_l = kernel.vector_lane if l == 1 else min(kernel.vector_lane, l)
        else:
            _tile_l = kernel.vector_lane
        tile_candidates = [[_tile_l, _tile_s, e]]

        for idx, (tile_l, tile_s, tile_e) in enumerate(tile_candidates):
            subtile_l = tile_l if (tile_l < kernel.vector_lane) or n_prologue_node else kernel.vector_lane
            subtile_s = tile_s # if (tile_s < kernel.vector_lane) or prologue_nodes else kernel.vector_lane
            subtile_e = tile_e # if (tile_e < kernel.vector_lane) or prologue_nodes else kernel.vector_lane

            tile_candidates[idx] = tile_l,tile_s,tile_e,subtile_l,subtile_s,subtile_e

        return tile_candidates

