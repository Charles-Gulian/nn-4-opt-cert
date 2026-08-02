"""Emit a LaTeX table describing the AC-OPF test systems.

For each IEEE/PEGASE case used in the AC-OPF experiments, loads the network
(problems/acopf/network.py:load_network) and reports: number of buses, lines
(branches), generators, the nominal TOTAL active demand (sum of P_D over all
buses, in MW), and the min/max total demand achievable under our two-level
uniform sampling scheme. Writes results/acopf_case_table.tex and prints the
raw numbers.

The total demand under sampling is  alpha * sum_i(eta_i * P_D0_i), with
alpha ~ U(ALPHA_MIN, ALPHA_MAX) system-wide and eta_i ~ U(1-ETA, 1+ETA) per
bus. Since the eta_i are independent, the extreme total demands are attained
when alpha and every eta_i hit the same bound simultaneously, giving
    min total = ALPHA_MIN * (1 - ETA) * S0,
    max total = ALPHA_MAX * (1 + ETA) * S0,
where S0 is the nominal total demand. Bounds are read from
problems/acopf/generate_data.py so they track the actual generation defaults.

Usage:
    /opt/anaconda3/envs/nn4opt/bin/python scripts/emit_acopf_case_table.py
"""

import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems.acopf.network import load_network
from problems.acopf.generate_data import (
    DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX, DEFAULT_ETA_RANGE,
)

CASES = ["case9", "case14", "case39", "case89pegase",
         "case118", "case300", "case1354pegase", "case2869pegase"]

# Total-demand multipliers under the sampling scheme (see module docstring).
MULT_MIN = DEFAULT_ALPHA_MIN * (1 - DEFAULT_ETA_RANGE)   # 0.6 * 0.7 = 0.42
MULT_MAX = DEFAULT_ALPHA_MAX * (1 + DEFAULT_ETA_RANGE)   # 1.2 * 1.3 = 1.56

OUT = PROJECT_ROOT / "results" / "acopf_case_table.tex"


def case_stats(case):
    _, nd = load_network(case)
    n_lines = len(nd.branch_from)
    pd_nom = np.asarray(nd.pd_nominal, dtype=float)
    total = float(pd_nom.sum())
    return {
        "buses": nd.n_buses,
        "lines": n_lines,
        "gens": nd.n_gens,
        "load_total": total,
        "load_min": MULT_MIN * total,
        "load_max": MULT_MAX * total,
    }


def main():
    print(f"Sampling multipliers on total demand: "
          f"[{MULT_MIN:.3g}, {MULT_MAX:.3g}] x nominal "
          f"(alpha in [{DEFAULT_ALPHA_MIN}, {DEFAULT_ALPHA_MAX}], "
          f"eta in [{1-DEFAULT_ETA_RANGE}, {1+DEFAULT_ETA_RANGE}])\n", flush=True)
    rows = []
    for case in CASES:
        s = case_stats(case)
        rows.append((case, s))
        print(f"{case:18s} buses={s['buses']:5d} lines={s['lines']:5d} "
              f"gens={s['gens']:4d}  total P_D (MW) nominal={s['load_total']:9.1f} "
              f"min={s['load_min']:9.1f} max={s['load_max']:9.1f}", flush=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AC-OPF test systems used in our experiments. The nominal "
        r"total active demand $P_D$ (summed over all buses, in MW) sets the "
        r"reference operating point; the min/max columns give the range of "
        r"total demand spanned by our two-level uniform sampling scheme "
        r"($\alpha \in [0.6, 1.2]$ system-wide, $\eta \in [0.7, 1.3]$ per bus). "
        r"Reactive demand $Q_D$ is scaled analogously.}",
        r"\label{tab:acopf-cases}",
        r"\begin{tabular}{l rrr rrr}",
        r"\toprule",
        r"System & Buses & Lines & Gens & \multicolumn{3}{c}{Total $P_D$ (MW)} \\",
        r"\cmidrule(lr){5-7}",
        r" & & & & nominal & min & max \\",
        r"\midrule",
    ]
    for case, s in rows:
        lines.append(
            f"\\texttt{{{case}}} & {s['buses']} & {s['lines']} & {s['gens']} & "
            f"{s['load_total']:.0f} & {s['load_min']:.0f} & {s['load_max']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
