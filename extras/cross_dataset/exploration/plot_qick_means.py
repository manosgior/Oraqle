import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = np.load(sys.argv[1]); out = sys.argv[2]
dt = 2500/770; t = np.arange(770)*dt
m0, m1 = d["m0"], d["m1"]
from numpy import convolve
def sm(x, k=9): return convolve(x, np.ones(k)/k, mode="same")
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].plot(t, sm(m0.real), label="Re <x|0>"); ax[0].plot(t, sm(m0.imag), label="Im <x|0>")
ax[0].plot(t, sm(m1.real), label="Re <x|1>"); ax[0].plot(t, sm(m1.imag), label="Im <x|1>"); ax[0].legend(ncol=4); ax[0].set_ylabel("mean (ADC)")
ax[1].plot(t, np.abs(sm(m1.real-m0.real)+1j*sm(m1.imag-m0.imag))); ax[1].set_ylabel("|<x|1>-<x|0>|")
ax[2].plot(t, d["s0"], label="std|0"); ax[2].plot(t, d["s1"], label="std|1"); ax[2].legend(); ax[2].set_ylabel("noise"); ax[2].set_xlabel("t (ns)")
for a in ax: a.set_xlim(0, 2500); a.grid(alpha=.3)
ax[0].set_xlim(0,2500)
plt.tight_layout(); plt.savefig(out, dpi=80)
# zoom: first 400 ns
print("first samples |m0|,|m1| at ns:", [(round(t[i]), round(abs(sm(m0.real)[i]+1j*sm(m0.imag)[i])), round(abs(sm(m1.real)[i]+1j*sm(m1.imag)[i]))) for i in range(0,130,10)])
