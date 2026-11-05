import sys, time, numpy as np, torch
sys.path.insert(0, "/scripts"); sys.argv = ["x", "/in", "/tmp"]
import m2mix_fit2 as F, m2mix_core2 as C
torch.set_num_threads(2)
x, b, y, sig0, c0, c1 = F.load("test", 1, "1us")
d = C.Binned(x, b, sig0, (c0, c1)); t = time.time()
P0, LL0 = C.fit(C.init_M0(d), C.Spec(), d); print("M0", LL0, time.time() - t)
print("logits", P0["logits"])
P1, LL1 = C.fit(C.add_g2(P0, [1.5, 3.4], d), C.Spec(g2="shared"), d); print("M1", LL1, P1["mu2"], time.time() - t)
r = C.describe(P1, C.Spec(g2="shared"), d, LL1); print(r["weights"], r["G2"]["along_anchor_sep"], r["G2"]["perp_anchor_sep"], r["G2"]["collapsed"])
P0f, LL0f = C.fit(C.add_flex(P0, d), C.Spec(flex=True), d); print("M0f", LL0f, time.time() - t)
P1f, LL1f = C.fit(C.add_g2(P0f, P1["mu2"], d), C.Spec(flex=True, g2="shared"), d); print("M1f", LL1f, time.time() - t)
P2f, LL2f = C.fit(C.add_g3(P1f, [0.3, 1.0], d), C.Spec(flex=True, g2="shared2"), d, max_rounds=3); print("M2f", LL2f, time.time() - t)
r = C.describe(P2f, C.Spec(flex=True, g2="shared2"), d, LL2f); print(r["weights"], r["far"])
Pn, LLn = C.fit(C.drop_extra(P2f, "3" if r["far"] == "G2" else "2"), C.Spec(flex=True, g2="shared"), d, max_rounds=3); print("nofar", LLn)
