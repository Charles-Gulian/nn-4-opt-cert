"""Tau-sweep certification analysis for the correlated-mu knapsack (MAX problem).

MAX-sense certificate (only maximization problem in the suite; signs flipped vs the
min problems): UB = vhat + q_alpha is a (1-alpha) UPPER bound on v*, where
q_alpha = conformal_offset(v_r - vhat, 1-alpha) = 99th-pctile upper offset (== -q01
in the paper's over-prediction notation). Certify f as delta-optimal iff
    vhat + q_alpha - f <= delta   <=>   (vhat - f) <= delta - q_alpha =: tau.
So the SINGLE deployed cutoff is tau on the score  s = vhat - f  (predicted
suboptimality of the local solver): certify iff s <= tau. A given (alpha, delta)
maps to tau = delta - q_alpha; conversely a chosen tau is realized by any (alpha,
delta) with delta = tau + q_alpha.

truth(delta) = (v* - f <= delta) is the ground-truth delta-optimality (uses exact v*).

Part A: sweep tau; at the canonical alpha=1% (delta = tau + q99) report
precision/recall/F1 and pick the best-F1 tau per model.
Part B: for a couple of chosen tau, tabulate (alpha, delta=tau+q_alpha) pairs, with
the self-consistent precision/recall (truth uses that pair's delta; FP guaranteed
<= alpha). q_alpha comes from the HELD-OUT calibration residuals (no test leakage).

Usage:
    .../python scripts/analyze_corr_certification.py
"""

import pathlib
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(PROJECT_ROOT))
from nn.metrics import conformal_offset

O = pathlib.Path("/private/tmp/claude-501/-Users-charlesgulian-Desktop-Projects-optimal-rolling-blackout/22e95144-1b5d-4142-af66-ceb1b4ad9159/scratchpad")
CELLS = [("exact 20k", "corr_exact_16k"), ("exact 80k", "corr_exact_64k"),
         ("relax 20k", "corr_relax_16k"), ("relax 80k", "corr_relax_64k")]
ALPHAS = [0.005, 0.01, 0.05, 0.10, 0.20]


def confusion(cert, truth):
    TP = int(np.sum(cert & truth)); FP = int(np.sum(cert & ~truth))
    TN = int(np.sum(~cert & ~truth)); FN = int(np.sum(~cert & truth))
    prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return TP, FP, TN, FN, prec, rec, f1


def load(tag):
    d = np.load(O / f"{tag}_data.npz"); r = np.load(O / f"{tag}.npz")
    return dict(vstar=d["vstar_te"], f=d["f_te"], vhat=r["vhat_te"],
                sv=float(r["sv"]), q99=float(r["q99"]), resid_cal=r["resid_cal"])


def main():
    for name, tag in CELLS:
        m = load(tag)
        s = m["vhat"] - m["f"]                       # score; certify iff s <= tau
        gap = m["vstar"] - m["f"]                     # true suboptimality
        sv, q99 = m["sv"], m["q99"]

        # ---- Part A: tau sweep at alpha=1% (delta = tau + q99) ----
        print(f"\n================ {name}  (std={sv:.2f}, q99={q99:.3f}={q99/sv:.3f}std) ================")
        print("Part A: sweep tau at alpha=1%  (delta = tau + q99); certify iff (vhat-f) <= tau")
        print(f"{'tau/std':>8} {'delta/std':>9} {'cert':>5} {'TP':>5} {'FP':>4} {'FN':>5} "
              f"{'prec':>6} {'recall':>7} {'F1':>6}")
        best = None
        taus = np.linspace(-0.02 * sv, 0.16 * sv, 46)
        for tau in taus:
            delta = tau + q99
            if delta < 0:
                continue
            cert = s <= tau
            truth = gap <= delta
            TP, FP, TN, FN, prec, rec, f1 = confusion(cert, truth)
            if best is None or f1 > best[0]:
                best = (f1, tau, delta, prec, rec, TP, FP, FN)
            if abs(round((tau / sv) / 0.02) * 0.02 - tau / sv) < 1e-9 or True:
                pass
        # print a coarse grid (every ~0.02 std) for readability
        for tau in np.arange(0.0, 0.161, 0.02) * sv:
            delta = tau + q99
            cert = s <= tau; truth = gap <= delta
            TP, FP, TN, FN, prec, rec, f1 = confusion(cert, truth)
            print(f"{tau/sv:8.3f} {delta/sv:9.3f} {100*cert.mean():4.0f}% {TP:5d} {FP:4d} "
                  f"{FN:5d} {prec:6.3f} {rec:7.3f} {f1:6.3f}")
        f1b, taub, deltab, precb, recb, TPb, FPb, FNb = best
        print(f"  best-F1 @ tau/std={taub/sv:.3f} (delta/std={deltab/sv:.3f}): "
              f"F1={f1b:.3f} prec={precb:.3f} recall={recb:.3f} (TP={TPb} FP={FPb} FN={FNb})")

        # ---- Part B: (alpha, delta) pairs for two chosen tau ----
        print("Part B: (alpha, delta) pairs realizing a chosen tau  "
              "[q_alpha from held-out calibration]")
        for label, tau in [("best-F1 tau", taub), ("high-prec tau=0", 0.0)]:
            print(f"  --- {label}: tau/std={tau/sv:.3f} "
                  f"(certified set fixed: {100*(s<=tau).mean():.1f}% of instances) ---")
            print(f"    {'alpha':>6} {'q_alpha/std':>11} {'delta/std':>9} "
                  f"{'prec':>6} {'recall':>7} {'FP%':>6} {'trueFPR<=a?':>11}")
            cert = s <= tau
            for a in ALPHAS:
                qa = conformal_offset(m["resid_cal"], 1 - a)   # upper offset at level 1-a
                delta = tau + qa
                truth = gap <= delta
                TP, FP, TN, FN, prec, rec, f1 = confusion(cert, truth)
                fp_rate = FP / len(s)
                ok = "yes" if fp_rate <= a else "NO"
                print(f"    {a:6.3f} {qa/sv:11.3f} {delta/sv:9.3f} {prec:6.3f} {rec:7.3f} "
                      f"{100*fp_rate:5.2f}% {ok:>11}")


if __name__ == "__main__":
    main()
