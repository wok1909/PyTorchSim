import torch, torch.nn as nn, math, os, sys, argparse
base = os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim")
sys.path.insert(0, base)
from Simulator.simulator import TOGSimulator
device = torch.device("npu:0")
CAUSAL = False


class GQAPrefill(nn.Module):
    """Naive causal GQA prefill attention (explicit ops, no flash).
    q: [H_q, S, D]   k,v: [H_kv, S, D]   (GQA: H_q = G * H_kv)
    """
    def forward(self, q, k, v, scale):
        H_q, S, D = q.shape
        G = H_q // k.shape[0]
        k = k.repeat_interleave(G, dim=0)              # [H_q, S, D]
        v = v.repeat_interleave(G, dim=0)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale          # [H_q, S, S]
        if CAUSAL:
            mask = torch.triu(torch.full((S, S), float("-inf"),
                                         device=q.device, dtype=scores.dtype), diagonal=1)
            scores = scores + mask
        p = torch.softmax(scores.float(), dim=-1).to(q.dtype)      # softmax over keys
        return torch.bmm(p, v)                                     # [H_q, S, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=5)       # LLAMA4_TP8: 5 q heads / 1 kv head
    ap.add_argument("--kv_heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--causal", type=int, default=0)
    a = ap.parse_args()
    global CAUSAL; CAUSAL = bool(a.causal)
    dt = torch.float16
    scale = 1.0 / math.sqrt(a.dim)
    q = torch.randn(a.heads, a.seq, a.dim, dtype=dt).to(device)
    k = torch.randn(a.kv_heads, a.seq, a.dim, dtype=dt).to(device)
    v = torch.randn(a.kv_heads, a.seq, a.dim, dtype=dt).to(device)
    comp = torch.compile(GQAPrefill().to(device), dynamic=False)
    with torch.no_grad(), TOGSimulator():
        out = comp(q, k, v, scale)
        torch.npu.synchronize()
    print(f"PREFILL_DONE S={a.seq} Hq={a.heads} Hkv={a.kv_heads} D={a.dim} out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
