"""CPU-only correctness check for the tiled causal flash-prefill logic.
No simulator / npu — just verifies the tiling math matches torch SDPA.
"""
import torch, torch.nn.functional as F, math


def flash_prefill(q, k, v, scale, Tq, Tk, causal):
    H_kv, G, Lq, D = q.shape
    S = k.shape[1]
    B = H_kv * G
    n_q = (Lq + Tq - 1) // Tq
    n_k = (S + Tk - 1) // Tk
    q_flat = q.reshape(B, Lq, D)
    k_rep = k.unsqueeze(1).expand(H_kv, G, S, D).reshape(B, S, D)
    v_rep = v.unsqueeze(1).expand(H_kv, G, S, D).reshape(B, S, D)
    # constant causal bias for the diagonal block (square aligned tiles)
    tri_bias = torch.triu(torch.full((Tq, Tk), -1e9), diagonal=1) if causal else None
    out_blocks = []
    for i in range(n_q):
        q0 = i * Tq; q1 = min(q0 + Tq, Lq); tq = q1 - q0
        q_i = q_flat[:, q0:q1, :]
        m_i = torch.full((B, tq, 1), float("-inf"))
        l_i = torch.zeros((B, tq, 1))
        acc = torch.zeros((B, tq, D))
        j_max = (q1 - 1) // Tk if causal else n_k - 1
        for j in range(j_max + 1):
            k0 = j * Tk; k1 = min(k0 + Tk, S); tk = k1 - k0
            k_j = k_rep[:, k0:k1, :]; v_j = v_rep[:, k0:k1, :]
            s = (torch.bmm(q_i, k_j.transpose(1, 2)) * scale).float()
            if causal and (k1 - 1) >= q0:
                s = s + tri_bias[:tq, :tk]
            m_new = torch.maximum(m_i, s.amax(-1, keepdim=True))
            p = (s - m_new).exp()
            alpha = (m_i - m_new).exp()
            l_i = l_i * alpha + p.sum(-1, keepdim=True)
            acc = acc * alpha + torch.bmm(p.to(q.dtype), v_j).float()
            m_i = m_new
        out_blocks.append((acc / l_i.clamp_min(1e-20)).to(q.dtype))
    return torch.cat(out_blocks, 1).view(H_kv, G, Lq, D)


def ref(q, k, v, scale, causal):
    H_kv, G, Lq, D = q.shape; S = k.shape[1]
    out = F.scaled_dot_product_attention(
        q.reshape(1, H_kv * G, Lq, D), k.reshape(1, H_kv, S, D), v.reshape(1, H_kv, S, D),
        attn_mask=None, dropout_p=0.0, is_causal=causal, enable_gqa=True, scale=scale)
    return out.reshape(H_kv, G, Lq, D)


def run(H_kv, G, S, D, Tq, Tk, causal, dtype=torch.float16):
    torch.manual_seed(0)
    q = torch.randn(H_kv, G, S, D, dtype=dtype)
    k = torch.randn(H_kv, S, D, dtype=dtype)
    v = torch.randn(H_kv, S, D, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    o1 = flash_prefill(q, k, v, scale, Tq, Tk, causal)
    o2 = ref(q, k, v, scale, causal)
    d = (o1.float() - o2.float()).abs().max().item()
    tag = f"Hkv={H_kv} G={G} S={S} D={D} Tq={Tq} Tk={Tk} causal={causal}"
    print(f"  max|flash-sdpa|={d:.6f}  {'PASS' if d < 0.05 else 'FAIL'}   [{tag}]")
    return d < 0.05


if __name__ == "__main__":
    ok = True
    # LLAMA4_TP8 shape, various sizes / tilings / causal+noncausal
    ok &= run(1, 5, 512, 128, 512, 512, True)     # single block
    ok &= run(1, 5, 2048, 128, 512, 512, True)    # 4x4 blocks, causal
    ok &= run(1, 5, 2048, 128, 512, 512, False)   # non-causal
    ok &= run(1, 5, 2048, 128, 256, 256, True)    # smaller square tiles (8x8 blocks)
    ok &= run(1, 5, 1536, 128, 512, 512, True)    # non-power-of-2 ragged (3 blocks)
    ok &= run(1, 5, 1000, 128, 512, 512, True)    # ragged last block (Lq not mult of Tq)
    ok &= run(2, 4, 1024, 64, 256, 256, True)     # multi kv-head, D=64
    ok &= run(1, 5, 2048, 128, 512, 1024, False)  # asymmetric tiles (non-causal only)
    print("ALL PASS" if ok else "SOME FAILED")
