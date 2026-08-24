"""Emit the paper's three results tables as LaTeX (booktabs + multirow).

Reads results/table_{offline,online,timing}.csv (written by
compute_final_tables.py) and writes results/tables_generated.tex,
paste-able directly into NN4OPT_TMLR/main-rewrite.tex in place of the
existing \\begin{table}...\\end{table} blocks for tab:offline/tab:online/
tab:timing (labels match, so in-text \\ref's keep working unchanged).

Column format (offline/online), approved 2026-08-24: drop the old grouped
headers (Data Generation/Convexification/Training/Calibration/Deployment)
and the old std/pct-infeasible/relax-gap/opt-gap columns, in favor of the
paper's own v_r/v/f notation directly:
    Table 2 (offline): v_r(theta) | v(theta) | f(x_hat;theta) | MAE | q_99
    Table 3 (online):  delta | TP | FP | TN | FN
Table 4 (timing) is unchanged from the original format.

Usage:  python scripts/emit_latex_tables.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

# Rows to include, in display order (knapsack: n=20000 correlated-mu rows
# only, per the "drop 80k from the main tables for now" decision).
KEEP = [
    ("QCQP", "Shor", None),
    ("Inverse Kinematics", "Shor", None),
    ("Inverse Kinematics", "Lasserre-2", None),
    ("MIMO Detection", "Shor", None),
    ("AC-OPF", "SOCP", "case9"), ("AC-OPF", "SOCP", "case14"),
    ("AC-OPF", "SOCP", "case39"), ("AC-OPF", "SOCP", "case89pegase"),
    ("AC-OPF", "SOCP", "case118"), ("AC-OPF", "SOCP", "case300"),
    ("AC-OPF", "SOCP", "case1354pegase"), ("AC-OPF", "SOCP", "case2869pegase"),
    ("AC-OPF", "SDP", "case9"), ("AC-OPF", "SDP", "case14"),
    ("AC-OPF", "SDP", "case39"), ("AC-OPF", "SDP", "case89pegase"),
    ("AC-OPF", "SDP", "case118"), ("AC-OPF", "SDP", "case300"),
    ("AC-OPF", "SDP", "case1354pegase"),
    ("Robust Knapsack", "Exact", "n=20000"), ("Robust Knapsack", "SOCP", "n=20000"),
]
RENAME_EXP = {"Robust Knapsack (corr.)": "Robust Knapsack"}
DISPLAY_CASE = {("Robust Knapsack", "Exact", "n=20000"): "n=25",
                 ("Robust Knapsack", "SOCP", "n=20000"): "n=25"}
SCI_THRESHOLD = 1e-3   # |x| below this -> scientific notation (e.g. IK's q_99)


def num(x, blank="--"):
    """Scale-aware numeric formatting for a table cell."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return blank
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 1:
        return f"{x:.3g}"
    if ax < SCI_THRESHOLD:
        return f"{x:.1e}"
    return f"{x:.2g}"


def sec(x):
    """Solve time in s / ms (unchanged from the original timing table)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x*1e3:.3g} ms" if x < 1 else f"{x:.3g} s"


def _texcase(c):
    return "" if not c else rf"\texttt{{{c}}}"


def _load(name):
    df = pd.read_csv(RESULTS / f"table_{name}.csv", keep_default_na=True, na_values=[""])
    df["relaxation"] = df["relaxation"].fillna("None").astype(str)
    df["experiment"] = df["experiment"].replace(RENAME_EXP)
    df["case"] = df["case"].fillna("")
    rows = []
    for exp, relax, case in KEEP:
        sub = df[(df["experiment"] == exp) & (df["relaxation"] == relax)
                 & (df["case"].fillna("") == (case or ""))]
        if len(sub) != 1:
            print(f"  MISSING/dup in {name}: {exp}/{relax}/{case} ({len(sub)} matches)",
                  file=sys.stderr)
            continue
        r = sub.iloc[0].to_dict()
        r["case"] = DISPLAY_CASE.get((exp, relax, case), case)
        rows.append(r)
    return rows


def _grouped_body(rows, cell_fn):
    out = []
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["experiment"] == rows[i]["experiment"]:
            j += 1
        for k in range(i, j):
            r = rows[k]
            exp_cell = rf"\multirow{{{j-i}}}{{*}}{{{r['experiment']}}}" if k == i else ""
            out.append("  " + " & ".join(
                [exp_cell, r["relaxation"], _texcase(r["case"])] + cell_fn(r)) + r" \\")
        out.append(r"\midrule")
        i = j
    if out and out[-1] == r"\midrule":
        out.pop()
    return "\n".join(out)


def offline_table(rows):
    def cells(r):
        # AC-OPF is nonconvex and generally intractable to solve exactly, so
        # v(theta) is NEVER genuinely available there (only relaxation
        # values of varying tightness stand in for it) -- blank rather than
        # show a relaxation value under a v(theta) header.
        vtrue_cell = "--" if r["experiment"] == "AC-OPF" else num(r["mean_vtrue"])
        return [num(r["mean_vr"]), vtrue_cell, num(r["mean_f"]),
                num(r["mae"]), num(r["q_offset"])]
    return rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Offline results, averaged over the four folds. ${{}}^\ast$AC-OPF is nonconvex and
generally intractable to solve exactly, so genuine $v(\bm\theta)$ is never available there
(only relaxation values of varying tightness); shown as --.}}
\label{{tab:offline}}
\begin{{tabular}}{{lll rrr rr}}
\toprule
\textbf{{Experiment}} & \textbf{{Relaxation}} & \textbf{{Case}} &
$\bar v_r(\bm\theta)$ & $\bar v(\bm\theta)^\ast$ & $\bar f(\hat{{\mathbf{{x}}}};\bm\theta)$ &
MAE & $q_{{99}}$ \\
\midrule
{_grouped_body(rows, cells)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def online_table(rows):
    def cells(r):
        return [num(r["delta"]), f"{int(round(r['TP']))}", f"{int(round(r['FP']))}",
                f"{int(round(r['TN']))}", f"{int(round(r['FN']))}"]
    return rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Online (deployment) results at $\delta = 3\%\cdot\mathrm{{std}}(v(\bm{{\theta}}))$ and
$\alpha = 1\%$ (offset $q_{{99}}$), averaged over the four folds.}}
\label{{tab:online}}
\begin{{tabular}}{{lll r rrrr}}
\toprule
\textbf{{Experiment}} & \textbf{{Relaxation}} & \textbf{{Case}} &
$\delta$ & TP & FP & TN & FN \\
\midrule
{_grouped_body(rows, cells)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def timing_table(rows):
    def cells(r):
        return [sec(r["relax_solve_s"]), sec(r["local_solve_s"])]
    return rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Mean solve times: the convex relaxation (offline) vs.\ the fast local solver.}}
\label{{tab:timing}}
\begin{{tabular}}{{lll rr}}
\toprule
\textbf{{Experiment}} & \textbf{{Relaxation}} & \textbf{{Case}} &
Relaxation Solve & Local Solve \\
\midrule
{_grouped_body(rows, cells)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main():
    off_rows = _load("offline")
    onl_rows = _load("online")
    tim_rows = _load("timing")
    tex = "\n\n".join([offline_table(off_rows), online_table(onl_rows),
                       timing_table(tim_rows)])
    out = RESULTS / "tables_generated.tex"
    out.write_text(tex + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
