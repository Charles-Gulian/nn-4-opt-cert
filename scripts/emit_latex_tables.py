"""Emit the three consolidated results tables as LaTeX (booktabs + multirow),
matching table mockup.xlsx. Reads results/table_{offline,online,timing}.csv,
writes results/tables_generated.tex (\\input-able / paste-able into the paper).

Usage:  python scripts/emit_latex_tables.py
"""
import json
import pathlib
import numpy as np
import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
_meta_path = RESULTS / "table_meta.json"
META = json.loads(_meta_path.read_text()) if _meta_path.exists() else {
    "delta_frac": 0.01, "alpha": 0.01, "level": 0.99}
QL = int(round(META["level"] * 100))        # offset subscript, e.g. 95
DPCT = f"{META['delta_frac']*100:.0f}\\%"    # escaped for LaTeX
APCT = f"{META['alpha']*100:.0f}\\%"


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
    return f"{x:.2g}"


def sec(x):
    """Solve time in s / ms."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x*1e3:.3g} ms" if x < 1 else f"{x:.3g} s"


def _texcase(c):
    return "" if not c or (isinstance(c, float) and not np.isfinite(c)) else rf"\texttt{{{c}}}"


def _body(df, cell_fns):
    """Rows with \\multirow on the Experiment column + midrules between groups."""
    out = []
    exps = df["experiment"].tolist()
    # group consecutive identical experiments
    groups = []
    i = 0
    while i < len(exps):
        j = i
        while j < len(exps) and exps[j] == exps[i]:
            j += 1
        groups.append((i, j))
        i = j
    for (a, b) in groups:
        for r in range(a, b):
            row = df.iloc[r]
            exp_cell = (rf"\multirow{{{b-a}}}{{*}}{{{row['experiment']}}}" if r == a else "")
            case = _texcase(row.get("case", ""))
            cells = [fn(row) for fn in cell_fns]
            out.append("  " + " & ".join([exp_cell, row["relaxation"], case] + cells) + r" \\")
        out.append(r"\midrule")
    if out and out[-1] == r"\midrule":
        out.pop()
    return "\n".join(out)


def offline_table(df):
    fns = [lambda r: num(r["mean_v"]), lambda r: num(r["std_v"]),
           lambda r: num(r["mean_relax_gap"]),
           lambda r: num(r["mae"]), lambda r: num(r["q_offset"])]
    return rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Offline results (data generation, convexification, training, and split-conformal
calibration), averaged over the four folds. ${{}}^\ast$If $v(\bm{{\theta}})$ is intractable we
use the relaxation value $v_r(\bm{{\theta}})$ for the mean, std, and relaxation gap.}}
\label{{tab:offline}}
\begin{{tabular}}{{lll rr r r r}}
\toprule
& & & \multicolumn{{2}}{{c}}{{Data Generation}} & Convexification & Training & Calibration \\
\cmidrule(lr){{4-5}}\cmidrule(lr){{6-6}}\cmidrule(lr){{7-7}}\cmidrule(lr){{8-8}}
\textbf{{Experiment}} & \textbf{{Relaxation}} & \textbf{{Case}} &
Mean $v^\ast$ & Std $v^\ast$ & Relax.\ Gap$^\ast$ & MAE & $q_{{{QL}}}$ \\
\midrule
{_body(df, fns)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def online_table(df):
    fns = [lambda r: num(r["mean_opt_gap"]), lambda r: num(r["worst_opt_gap"]),
           lambda r: f"{int(r['TP'])}", lambda r: f"{int(r['FP'])}",
           lambda r: f"{int(r['TN'])}", lambda r: f"{int(r['FN'])}"]
    return rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Online (deployment) results at $\delta = {DPCT}\cdot\mathrm{{std}}(v(\bm{{\theta}}))$ and
$\alpha = {APCT}$ (offset $q_{{{QL}}}$), averaged over the four folds. ${{}}^\ast$Optimality gap uses
$v(\bm{{\theta}})$ where tractable, else $v_r(\bm{{\theta}})$.}}
\label{{tab:online}}
\begin{{tabular}}{{lll rr rrrr}}
\toprule
& & & \multicolumn{{6}}{{c}}{{Deployment}} \\
\cmidrule(lr){{4-9}}
\textbf{{Experiment}} & \textbf{{Relaxation}} & \textbf{{Case}} &
Mean Gap$^\ast$ & Worst Gap$^\ast$ & TP & FP & TN & FN \\
\midrule
{_body(df, fns)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def timing_table(df):
    fns = [lambda r: sec(r["relax_solve_s"]), lambda r: sec(r["local_solve_s"])]
    return rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Mean solve times: the convex relaxation (offline) vs.\ the fast local solver.
Higher-order relaxations are consistently slower.}}
\label{{tab:timing}}
\begin{{tabular}}{{lll rr}}
\toprule
\textbf{{Experiment}} & \textbf{{Relaxation}} & \textbf{{Case}} &
Relaxation Solve & Local Solve \\
\midrule
{_body(df, fns)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main():
    kw = dict(keep_default_na=True, na_values=[""])   # keep "None" as a string
    off = pd.read_csv(RESULTS / "table_offline.csv", **kw)
    onl = pd.read_csv(RESULTS / "table_online.csv", **kw)
    tim = pd.read_csv(RESULTS / "table_timing.csv", **kw)
    for df in (off, onl, tim):
        df["relaxation"] = df["relaxation"].fillna("None").astype(str)
    tex = "\n\n".join([offline_table(off), online_table(onl), timing_table(tim)])
    out = RESULTS / "tables_generated.tex"
    out.write_text(tex + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
