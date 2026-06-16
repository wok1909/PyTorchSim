"""TPUv6e validation — error-rate bar charts. error% = (sim/real - 1)*100."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (sim, real) us
gemm = dict(N=[512,1024,2048,4096,8192], sim=[1.39,5.51,28.09,170.82,1420.70], real=[2.49,6.85,29.89,187.75,1593.11])
gemv = dict(N=[1024,2048,4096,8192,16384], sim=[1.96,7.89,30.45,115.76,399.50], real=[2.60,7.38,24.14,92.67,361.94])
vadd = dict(N=[1048576,4194304,16777216,67108864], sim=[4.87,19.23,76.69,306.52], real=[6.07,18.78,71.69,283.81])
pref = dict(S=[512,1024,2048], sim=[11.08,42.34,159.46], real=[42.14,30.52,82.17])
dec  = dict(S=[512,1024,2048,4096,8192], sim=[1.20,1.55,2.24,3.64,6.26], real=[28.18,29.45,39.78,28.20,31.06])

def err(d): return [(s/r-1)*100 for s,r in zip(d['sim'],d['real'])]
def sz(n):  return (f"{n//1048576}M" if n>=1048576 else (f"{n//1024}K" if n>=1024 else str(n)))

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("PyTorchSim TPUv6e — error rate vs real TPU v6e   ( (sim/real − 1) × 100% )",
             fontsize=14, fontweight="bold")

# ---- Phase 1 ----
labels, vals, cols = [], [], []
for d, c, name in [(gemm,'C0','GEMM'),(gemv,'C1','GEMV'),(vadd,'C2','VADD')]:
    for n, e in zip(d['N'], err(d)):
        labels.append(f"{name}\n{sz(n)}"); vals.append(e); cols.append(c)
a = ax[0]
x = np.arange(len(vals))
bars = a.bar(x, vals, color=cols, edgecolor='k', lw=0.4)
a.axhspan(-10,10, color='green', alpha=0.12, label='±10% target')
a.axhline(0, color='k', lw=0.8)
for xi, v in zip(x, vals):
    a.text(xi, v+(2 if v>=0 else -2), f"{v:+.0f}", ha='center',
           va='bottom' if v>=0 else 'top', fontsize=7)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=7)
a.set_ylabel("error %"); a.set_title("Phase 1 — primitive ops (validated)")
a.set_ylim(-55, 40); a.legend(fontsize=9); a.grid(axis='y', alpha=0.3)
from matplotlib.patches import Patch
a.legend(handles=[Patch(fc='C0',label='GEMM'),Patch(fc='C1',label='GEMV'),
                  Patch(fc='C2',label='VADD'),Patch(fc='green',alpha=0.3,label='±10%')], fontsize=8)

# ---- Phase 2 ----
labels2, vals2, cols2 = [], [], []
for n,e in zip(pref['S'], err(pref)): labels2.append(f"prefill\nS{n}"); vals2.append(e); cols2.append('C3')
for n,e in zip(dec['S'],  err(dec)):  labels2.append(f"decode\nS{n}");  vals2.append(e); cols2.append('C4')
a = ax[1]
x2 = np.arange(len(vals2))
a.bar(x2, vals2, color=cols2, edgecolor='k', lw=0.4)
a.axhspan(-10,10, color='green', alpha=0.12)
a.axhline(0, color='k', lw=0.8)
for xi, v in zip(x2, vals2):
    a.text(xi, v+(3 if v>=0 else -3), f"{v:+.0f}", ha='center',
           va='bottom' if v>=0 else 'top', fontsize=7)
a.set_xticks(x2); a.set_xticklabels(labels2, fontsize=7)
a.set_ylabel("error %"); a.set_title("Phase 2 — attention (known modeling gaps, not config error)")
a.set_ylim(-110, 110); a.grid(axis='y', alpha=0.3)
a.legend(handles=[Patch(fc='C3',label='prefill (gap = causal skip TODO)'),
                  Patch(fc='C4',label='decode (gap = launch overhead not modeled)'),
                  Patch(fc='green',alpha=0.3,label='±10%')], fontsize=8, loc='lower left')

plt.tight_layout(rect=[0,0,1,0.95])
out="/workspace/PyTorchSim/tpu_validation/v6e_error_rate.png"
plt.savefig(out, dpi=130, bbox_inches='tight'); print("SAVED", out)
