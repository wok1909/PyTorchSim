import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import argparse
from Simulator.simulator import TOGSimulator
device = torch.device("npu:0")
# ─────────────────────────────────────────────────────────────────────────────
# Flash-Prefill style GQA prefill attention for multi-core NPU.
#
# Unlike decode (query = 1 token), prefill has a *full query sequence* of
# length Lq (== S for self-attention).  We therefore tile BOTH axes:
#
#     • query axis Lq  → independent blocks of size Tq  (fold heads into batch)
#     • key   axis S   → blocks of size Tk, walked with an online-softmax loop
#
# Causal masking is handled at *block granularity*:
#     • a key block entirely in the future of a query block   → skipped (no compute)
#     • the diagonal block (partial overlap)                  → small [Tq,Tk] mask
#     • a key block entirely in the past                      → no mask
#
# The largest intermediate ever materialized is a single [B, Tq, Tk] score tile,
# never the full [H_q, S, S] matrix.  This is the prefill analogue of the
# Flash-Decode tiling in GQADecodeOptimized.
# ─────────────────────────────────────────────────────────────────────────────


class GQAPrefillFlash(nn.Module):
    """Tiled causal flash-attention prefill for GQA.

    Input conventions
    ─────────────────
        q : [H_kv, G, Lq, D]  – Lq query tokens per (kv-head, query-head-in-group)
        k : [H_kv, S,  D]     – keys   (NOT pre-transposed)
        v : [H_kv, S,  D]     – values

    Tiling
    ──────
        Tq : query block size   (rows of the score tile)
        Tk : key   block size   (cols of the score tile)
        Heads (H_kv × G) are folded into the BMM batch dim B so all heads of a
        query block are issued as one batched BMM; the key loop (j) carries the
        online-softmax running max/sum.
    """

    def __init__(self, tile_q: int = 512, tile_k: int = 512, causal: bool = True):
        super().__init__()
        if causal and tile_q != tile_k:
            raise ValueError("causal path requires square tiles (tile_q == tile_k)")
        self.tile_q = tile_q
        self.tile_k = tile_k
        self.causal = causal
        # Causal additive bias for the diagonal block (q0 == k0, square tile):
        #   allowed (col <= row) → 0 ;  future (col > row) → -1e9
        # Precomputed constant → no boolean/masked_fill in the compiled graph
        # (those trip the NPU inductor mask-buffer codegen path).
        if causal:
            tri = torch.triu(torch.full((tile_q, tile_k), -1e9, dtype=torch.float32), diagonal=1)
            self.register_buffer("tri_bias", tri)

    def forward(
        self,
        q: torch.Tensor,   # [H_kv, G, Lq, D]
        k: torch.Tensor,   # [H_kv, S, D]
        v: torch.Tensor,   # [H_kv, S, D]
        scale: float,
    ) -> torch.Tensor:
        H_kv, G_orig, Lq, D = q.shape
        _, S, _        = k.shape
        Tq             = self.tile_q
        Tk             = self.tile_k

        # ── Pad the folded batch B = H_kv*G to EVEN ──────────────────────────
        # The NPU codegen vector-tiles the batch dim by 2; an odd B makes the
        # last 2-wide group read a phantom head one element past the score
        # tensor → gem5 OOB abort.  Pad G with a zero head so B is even.
        G = G_orig
        if (H_kv * G) % 2 == 1:
            G = G_orig + 1
            q = F.pad(q, (0, 0, 0, 0, 0, G - G_orig))   # pad G dim with one zero head
        B = H_kv * G

        n_q = (Lq + Tq - 1) // Tq
        n_k = (S  + Tk - 1) // Tk

        # ── Fold heads into batch.  K/V are shared across G → expand. ─────────
        #   q_flat : [B, Lq, D]
        #   k_rep  : [B, S, D]   (each kv-head replicated G times)
        q_flat = q.reshape(B, Lq, D)
        k_rep  = k.unsqueeze(1).expand(H_kv, G, S, D).reshape(B, S, D)
        v_rep  = v.unsqueeze(1).expand(H_kv, G, S, D).reshape(B, S, D)

        out_blocks = []
        for i in range(n_q):
            q0 = i * Tq
            q1 = min(q0 + Tq, Lq)
            q_i = q_flat[:, q0:q1, :]                       # [B, tq, D]
            tq  = q1 - q0

            # Online-softmax running state for this query block
            m_i = torch.full((B, tq, 1), float("-inf"), dtype=torch.float32, device=q.device)
            l_i = torch.zeros((B, tq, 1), dtype=torch.float32, device=q.device)
            acc = torch.zeros((B, tq, D), dtype=torch.float32, device=q.device)

            # Highest key index this query block can attend to (causal)
            j_max = n_k - 1
            if self.causal:
                # last (largest) query row in this block = q1 - 1
                j_max = (q1 - 1) // Tk

            for j in range(j_max + 1):
                k0 = j * Tk
                k1 = min(k0 + Tk, S)
                tk = k1 - k0
                k_j = k_rep[:, k0:k1, :]                    # [B, tk, D]
                v_j = v_rep[:, k0:k1, :]                    # [B, tk, D]

                # QK^T  → [B, tq, tk]
                s = torch.bmm(q_i, k_j.transpose(1, 2)) * scale
                s = s.float()

                # Diagonal block (q0 == k0 for square aligned tiles) → add the
                # precomputed lower-triangular causal bias (constant add, no mask buffer)
                if self.causal and (k1 - 1) >= q0:
                    s = s + self.tri_bias[:tq, :tk]

                # Online softmax update
                m_new = torch.maximum(m_i, s.amax(dim=-1, keepdim=True))        # [B,tq,1]
                p     = (s - m_new).exp()                                       # [B,tq,tk]
                alpha = (m_i - m_new).exp()                                     # [B,tq,1]
                l_i   = l_i * alpha + p.sum(dim=-1, keepdim=True)
                # Keep the P@V matmul in its own buffer so the accumulator/cast
                # don't get fused into the matmul template epilogue (autotune).
                pv    = torch.bmm(p.to(q.dtype), v_j)                           # [B,tq,D]
                acc   = acc * alpha + pv.float()
                m_i   = m_new

            out_i = (acc / l_i.clamp_min(1e-20)).to(q.dtype)                    # [B,tq,D]
            out_blocks.append(out_i)

        out = torch.cat(out_blocks, dim=1)                  # [B, Lq, D]
        out = out.view(H_kv, G, Lq, D)
        return out[:, :G_orig].contiguous()                 # drop padded head(s)


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "LLAMA4_TP8": {
        "HEAD_DIM":     128,
        "NUM_HEADS":    5,    # = 40 total / TP8
        "NUM_KV_HEADS": 1,    # =  8 total / TP8
    },
    "QWEN3-235B_TP4": {
        "HEAD_DIM":     128,
        "NUM_HEADS": 16,
        "NUM_KV_HEADS": 1,
    },
    "GPT-OSS_TP8": {
        "HEAD_DIM":     64,
        "NUM_HEADS":  8,
        "NUM_KV_HEADS": 1,
    },
}


def _make_inputs(cfg, seq_len, dtype):
    H_kv  = cfg["NUM_KV_HEADS"]
    G     = cfg["NUM_HEADS"] // cfg["NUM_KV_HEADS"]
    D     = cfg["HEAD_DIM"]
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(H_kv, G, seq_len, D, dtype=dtype)
    k = torch.randn(H_kv, seq_len, D,    dtype=dtype)   # NOT pre-transposed
    v = torch.randn(H_kv, seq_len, D,    dtype=dtype)
    return q, k, v, scale


def _reference(q, k, v, scale, causal):
    """CPU eager reference via torch SDPA (math), GQA-aware.

    q [H_kv,G,Lq,D] → [1, H_kv*G, Lq, D];  k/v [H_kv,S,D] → [1, H_kv, S, D].
    """
    H_kv, G, Lq, D = q.shape
    S = k.shape[1]
    q_r = q.reshape(1, H_kv * G, Lq, D)
    k_r = k.reshape(1, H_kv, S, D)
    v_r = v.reshape(1, H_kv, S, D)
    out = F.scaled_dot_product_attention(
        q_r, k_r, v_r, attn_mask=None, dropout_p=0.0,
        is_causal=causal, enable_gqa=True, scale=scale,
    )
    return out.reshape(H_kv, G, Lq, D)


def test_gqa_prefill_flash(model, device, seq_len, tile_q, tile_k, causal):
    cfg   = MODEL_CONFIGS[model] if model is not None else MODEL_CONFIGS["LLAMA4_TP8"]
    dtype = torch.float16

    net = GQAPrefillFlash(tile_q=tile_q, tile_k=tile_k, causal=causal).eval()

    q, k, v, scale = _make_inputs(cfg, seq_len, dtype)

    # ── NPU run ──────────────────────────────────────────────────────────────
    net_dev  = net.to(device)
    compiled = torch.compile(net_dev, dynamic=False)
    q_dev, k_dev, v_dev = q.to(device), k.to(device), v.to(device)
    with torch.no_grad():
        with TOGSimulator():
            out_dev = compiled(q_dev, k_dev, v_dev, scale=scale)

    # ── CPU references ─────────────────────────────────────────────────────────
    with torch.no_grad():
        out_self = net.cpu()(q, k, v, scale=scale)                  # our own kernel on CPU
        out_lib  = _reference(q, k, v, scale, causal)               # torch SDPA reference

    max_diff_self = (out_dev.cpu() - out_self).abs().max().item()
    max_diff_lib  = (out_dev.cpu() - out_lib ).abs().max().item()
    self_vs_lib   = (out_self      - out_lib ).abs().max().item()

    print(f"[GQAPrefillFlash] model={model} seq_len={seq_len} "
          f"tile_q={tile_q} tile_k={tile_k} causal={causal}")
    print(f"  max |npu  - self_cpu| = {max_diff_self:.6f}")
    print(f"  max |npu  - sdpa_lib| = {max_diff_lib:.6f}")
    print(f"  max |self - sdpa_lib| = {self_vs_lib:.6f}")
    print(f"  npu  out max = {out_dev.cpu().abs().max().item():.6f}")
    print(f"  self out max = {out_self.abs().max().item():.6f}")
    print(f"  lib  out max = {out_lib.abs().max().item():.6f}")
    print("  PASS" if max_diff_lib < 0.05 else "  FAIL (diff too large)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test GQA Flash-Prefill attention")
    ap.add_argument("--model", type=str, default="LLAMA4_TP8", choices=MODEL_CONFIGS.keys())
    ap.add_argument("--context_length", type=int, default=2048)
    ap.add_argument("--tile_q", type=int, default=512)
    ap.add_argument("--tile_k", type=int, default=512)
    ap.add_argument("--causal", type=int, default=1)
    args = ap.parse_args()
    base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
    sys.path.append(base_dir)
    test_gqa_prefill_flash(
        model=args.model, device=device, seq_len=args.context_length,
        tile_q=args.tile_q, tile_k=args.tile_k, causal=bool(args.causal),
    )
