"""TPUv6e PyTorchSim validation — consolidated plots.
Phase 1 (GEMM/GEMV/VADD) + Phase 2 (attention prefill/decode), sim vs real.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Phase 1: primitive ops (sim fp16 vs real bf16), us ----
# sim = measured cycles / 3502 MHz (opt1 vlen-128, CMEM=0, fp16, functional-off);
# GEMM re-measured after the chunked zero-init fix (8192 previously crashed in autotune).
gemm = dict(N=[512,1024,2048,4096,8192], sim=[1.66,6.65,31.30,186.28,1264.45], real=[2.49,6.85,29.89,187.75,1593.11])
gemv = dict(N=[1024,2048,4096,8192,16384], sim=[2.05,7.83,31.75,115.77,399.51], real=[2.60,7.38,24.14,92.67,361.94])
vadd = dict(N=[1048576,4194304,16777216,67108864], sim=[4.87,19.24,76.70,306.53], real=[6.07,18.78,71.69,283.81])

# ---- Phase 2: attention (sim fp32 functional-off vs real bf16), B=1, us ----
pref = dict(S=[512,1024,2048], sim=[11.08,42.34,159.46], real=[42.14,30.52,82.17])
dec  = dict(S=[512,1024,2048,4096,8192], sim=[1.20,1.55,2.24,3.64,6.26], real=[28.18,29.45,39.78,28.20,31.06])

fig, ax = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("PyTorchSim TPUv6e  vs  real TPU v6e-1 — validation", fontsize=15, fontweight="bold")

# (1) Phase 1: sim vs real latency, log-log, y=x + ±10% band
a = ax[0,0]
for d, m, c, lbl in [(gemm,'o','C0','GEMM'),(gemv,'s','C1','GEMV'),(vadd,'^','C2','VADD')]:
    a.scatter(d['real'], d['sim'], marker=m, color=c, s=60, label=lbl, zorder=3)
lo, hi = 1, 2000
a.plot([lo,hi],[lo,hi],'k--',lw=1,label='sim=real')
a.fill_between([lo,hi],[lo*0.9,hi*0.9],[lo*1.1,hi*1.1],color='gray',alpha=0.2,label='±10%')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo,hi); a.set_ylim(lo,hi)
a.set_xlabel("real µs"); a.set_ylabel("sim µs")
a.set_title("Phase 1: primitive-op latency (sim vs real)")
a.legend(fontsize=8); a.grid(True, which='both', alpha=0.3)

# (2) Phase 1: sim/real ratio vs size, ±10% band
a = ax[0,1]
for d, m, c, lbl in [(gemm,'o','C0','GEMM'),(gemv,'s','C1','GEMV'),(vadd,'^','C2','VADD')]:
    ratio = np.array(d['sim'])/np.array(d['real'])
    a.plot(d['N'], ratio, marker=m, color=c, label=lbl)
a.axhspan(0.9,1.1,color='green',alpha=0.12,label='±10%')
a.axhline(1.0,color='k',lw=0.8)
a.set_xscale('log'); a.set_xlabel("problem size N"); a.set_ylabel("sim / real")
a.set_title("Phase 1: sim/real ratio (accuracy)")
a.set_ylim(0.4,1.4); a.legend(fontsize=8); a.grid(True, alpha=0.3)

# (3) Phase 2: prefill latency vs S
a = ax[1,0]
a.plot(pref['S'], pref['real'], 'o-', color='C3', label='real (bf16, causal)')
a.plot(pref['S'], pref['sim'],  's--', color='C0', label='sim (fp32, full S²)')
a.set_xscale('log',base=2); a.set_yscale('log')
a.set_xlabel("seq len S"); a.set_ylabel("µs")
a.set_title("Phase 2: PREFILL latency (B=1)")
a.legend(fontsize=9); a.grid(True, which='both', alpha=0.3)
a.annotate("~2× gap = causal\nfuture-tile skip\nnot yet implemented",
           xy=(2048,159.5), xytext=(560,250), fontsize=8,
           arrowprops=dict(arrowstyle='->',color='gray'))

# (4) Phase 2: decode latency vs S
a = ax[1,1]
a.plot(dec['S'], dec['real'], 'o-', color='C3', label='real (bf16 flash)')
a.plot(dec['S'], dec['sim'],  's--', color='C0', label='sim (fp16)')
a.set_xscale('log',base=2)
a.set_xlabel("KV-cache len S"); a.set_ylabel("µs")
a.set_title("Phase 2: DECODE latency (B=1)")
a.legend(fontsize=9); a.grid(True, alpha=0.3)
a.annotate("real flat ~30µs =\nkernel-launch overhead\n(sim not modeled)",
           xy=(4096,28.2), xytext=(600,15), fontsize=8,
           arrowprops=dict(arrowstyle='->',color='gray'))

plt.tight_layout(rect=[0,0,1,0.96])
out = "/workspace/PyTorchSim/tpu_validation/v6e_validation_plots.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print("SAVED", out)
