# XLU (Cross-Lane Unit) — research notes for TOGSim implementation

Collected 2026-06 for the purpose of modeling a cross-lane reduce/permute unit
in TOGSim (needed for flash-decode split-KV merge, and faithful to real TPU HW).

The point of this doc: real TPU TensorCores have a *separate* unit, the XLU, that
does cross-lane data movement + reduction. PyTorchSim/TOGSim currently models the
VPU as strictly per-lane, so it has no equivalent. This is the gap to fill.

---

## 1. What the XLU is

- Full name: **XLU = Cross-Lane Unit** (a.k.a. "Transpose-Reduction-Permute Unit").
- It is a **distinct functional unit**, sibling to the MXU, NOT part of the per-lane VPU.
- Per-core composition (TPU v5p, from the JAX scaling book):
  > "A single scalar core controls a VPU (consisting of 4096 ALUs), **4 MXUs, 2 XLUs**, and multiple DMA engines."
  - VPU 4096 ALUs = 8 sublane x 128 lane x 4 ALUs.
- The XLU itself is made of two sub-pieces (from Google's matmul patents):
  1. **Transpose Unit (XU)** — does a **128x128 transpose**.
  2. **Reduction & Permutation Unit** — does cross-lane reduce + permute/rotate.

## 2. Operations it supports

From the patents (US10621269 family "Performing matrix multiplication in hardware",
and US11966745B2 "Sparse SIMD cross-lane processing unit"):

- **permutation** — arbitrary lane->lane rearrangement
- **lane rotation** — cyclic shift across lanes
- **rotating permutation** — rotation + element permute combined
- **lane reduction** — aggregate (sum/max/...) across lanes
- **permuted lane reduction** — permute then reduce
- **segmented permuted lane reduction** — partition lanes into segments, reduce per segment
- **transpose** — 128x128 (the XU)

So it is NOT a single fixed behavior. It is a menu: move data sideways (permute/rotate),
combine data sideways (reduce), or both (segmented/permuted reduce). Broadcast is just a
permutation special case (one lane -> many).

## 3. How it works internally

- Built around a **crossbar / rotation network** that physically moves operands between lanes.
  From US11966745B2: "the XPU includes a network of individual processing cells, each cell
  processing data that passes through one or more data processing lanes through crossbar
  connections." Data flows through pipeline **stages**, each stage = processing cells + a crossbar;
  "the crossbar is configured to permute input values from each processing cell in the same stage
  according to a fixed pattern."
- A full cross-lane reduction is done as a **log-step tree**: rotate + add, repeated log2(N) times.
  From the JAX scaling book / Patrick Toulme VLIW trace:
  > "classic parallel reduction pattern uses **log2(n) steps — rotate by 4, 2, then 1** — to sum 8 sublanes into a single scalar."
  - For the lane axis (128 lanes) this is ceil(log2(128)) = 7 rotate+add steps.
  - After the tree completes, every lane holds the result (all-reduce style => reduce+broadcast for free).
- The lane-axis reduction is often implemented as **transpose -> tree-reduce**: the compiler
  transposes so the reduce axis lands on the (cheap) sublane axis, then does the log-step rotate+add.
  From the VLIW trace: "After transposing, a tree reduction sums the 16 lanes."

## 4. Cheap vs expensive — the key asymmetry

This is the single most important fact for cost modeling:

- **Sublane-axis (8) reduction = cheap.** The VPU has an intra-lane shuffle that "can roll along
  the axis of size 8 in about a cycle" — shuffle by 4,2,1 + 3 elementwise sums. No XLU needed.
- **Lane-axis (128) reduction = expensive.** Must go through the XLU, which the scaling book calls
  **"slow and fairly expensive."** The HE/ASIC paper (arXiv 2501.07047): the XLU "consumes
  **non-hidden layout reordering and reduction latency**" — i.e. its cost is NOT hidden behind compute,
  unlike the MXU's local transpose unit which IS pipelined/hidden.
- Pallas TPU docs phrase the same asymmetry at the software level:
  > "Reductions over the last array dimension are generally the slowest. Reductions over the second
  > last dimension are faster, but still slower than over the leading dimensions."
  > "Broadcasting along all but the two trailing dimensions is always supported and free. Broadcasting
  > along the second to last dimension is slower, while broadcasting along the last dimension is the slowest."

## 5. Software / compiler view (Pallas + Mosaic)

- Pallas (TPU) lowers through **Mosaic**, which decides tiling and emits the XLU ops.
- Pallas TPU docs: TPUs have an "XLU unit" supporting "matrix transpositions and permutes" as
  **asynchronous** operations that execute in the background.
- Transposes of the **last two axes** may be **fused into the matmul** (the MXU's local transpose unit),
  which hides their latency; standalone cross-lane reductions through the XLU are the ones that cost.

## 6. Latency data — what is and isn't public

- **No public per-cycle latency number** for the TPU XLU lane reduction. All sources call it
  "slow / expensive / non-hidden" but give no cycle count.
- Closest concrete numbers come from the **sparse cross-lane unit patent** (US11966745B2), which is a
  *related* design, not the exact TPU XLU: vector sort = 6 cycles, prefix sum = 4 cycles, compact = 2 cycles.
- Practical implication: we'll have to **model it from first principles** = log-step tree:
  `latency ~= ceil(log2(num_lanes)) * (rotate + add) cost`, plus a fixed crossbar traversal overhead,
  and mark it as non-overlapped (serializes with the VPU pipeline). This is defensible because it's
  exactly the documented rotate-by-4/2/1 mechanism, not an arbitrary fudge.

## 7. Implications for TOGSim implementation

1. Add a **new functional-unit / op type** for cross-lane reduce (and optionally transpose/permute) —
   sibling to the existing VECTOR_UNIT / MXU, NOT a VPU instruction. (The earlier `vfredusum` attempt
   failed precisely because it tried to express cross-lane reduce as a per-lane VPU op.)
2. **Cost model**: lane-axis reduce = `ceil(log2(active_lanes))` rotate+add steps, non-hidden.
   Sublane-axis reduce stays cheap (~1 cycle path) and can remain VPU-modeled.
3. **MLIR exposure**: flash-decode's split-KV partial-softmax merge needs a cross-lane all-reduce;
   expose it as a dedicated op that lowers to the new XLU unit rather than a vector reduction intrinsic.
4. Number per core (`2 XLUs` on v5p) -> a config knob, analogous to `num_systolic_array_per_core`.

---

## Sources (organized)

### Authoritative architecture overviews
- JAX scaling book, "How to Think About TPUs" — VPU/XLU structure, 2 XLUs/core, sublane-vs-lane cost,
  log-step reduction: https://jax-ml.github.io/scaling-book/tpus/
- Ironwood paper (arXiv 2606.15870) — "transposes, row reductions, or column permutations" done by
  "other units"; VPU registers 8x128 -> 16x256. (local: /tmp/ironwood.txt)
- Inside the Ironwood TPU codesigned AI stack (Google Cloud blog):
  https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack

### Compiler / software-facing (how XLU ops are emitted)
- Pallas TPU details (cross-lane ops, transpose fusion, reduction/broadcast cost asymmetry):
  https://docs.jax.dev/en/latest/pallas/tpu/details.html
- Patrick Toulme, "From JAX to VLIW" (actual VLIW trace of transpose -> tree reduction):
  https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation
- "The Rise of Pallas" (Mosaic backend overview):
  https://towardsdatascience.com/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a/

### Patents (hardware structure — the most detailed)
- US10621269 (+ family US10831862, US11989258, US10698974) "Performing matrix multiplication in
  hardware" — XLU = transpose unit (XU, 128x128) + reduction/permutation unit; lists permutation,
  lane rotation, rotating permutation, lane reduction, permuted lane reduction, segmented permuted
  lane reduction: https://patents.google.com/patent/US10621269
- US11966745B2 "Sparse SIMD cross-lane processing unit" — crossbar/stage network, per-op cycle counts
  for the sparse variant: https://patents.google.com/patent/US11966745B2/en
- US10209989 "Accelerated interlane vector reduction instructions":
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10209989
- US10970081 "Stream processor with decoupled crossbar for cross lane operations":
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10970081
- Low-latency matrix multiply unit family: US11907330, US10970362, US11599601

### Quantitative / modeling references
- "Leveraging ASIC AI Chips for Homomorphic Encryption" (arXiv 2501.07047) — XLU "consumes non-hidden
  layout reordering and reduction latency"; local MXU transpose is hidden: https://arxiv.org/html/2501.07047v2

### Background
- TPU architecture (QSysArch): https://qsysarch.com/posts/tpu-architecture/
- Google TPU Architecture: 7 Generations (Introl): https://introl.com/blog/google-tpu-architecture-complete-guide-7-generations
