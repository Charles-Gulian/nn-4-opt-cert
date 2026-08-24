"""Emit the paper's one combined results table as LaTeX (booktabs + multirow,
landscape via pdflscape).

Reads results/table_{offline,online,timing}.csv (written by
compute_final_tables.py) and writes results/tables_generated.tex, paste-able
directly into NN4OPT_TMLR/main-rewrite.tex in place of the existing
\\begin{landscape}...\\end{landscape} block for tab:combined (label matches, so
in-text \\ref's keep working unchanged).

Column format, approved 2026-08-24 (offline+online+timing merged into one
table): v_r(theta) | v(theta) | f(x_hat;theta) | MAE | q_99 | delta | TP | FP
| TN | FN | Relaxation Solve | Local Solve, dropping the old grouped headers
and the old std/pct-infeasible/relax-gap/opt-gap columns in favor of the
paper's own v_r/v/f notation directly (gaps are visible by eye from the raw
values).

NOTE on case2869pegase's chordal SDP: it has no offline/online row (no
certification was ever computed for it -- see the paper's dagger footnote),
so it is NOT in KEEP and must be spliced in as a timing-only row by hand when
pasting into the document (as already done in main-rewrite.tex).

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
    """Solve time in s / ms."""
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


def _grouped_body(off_rows, onl_rows, tim_rows):
    out = []
    i = 0
    n = len(off_rows)
    while i < n:
        j = i
        while j < n and off_rows[j]["experiment"] == off_rows[i]["experiment"]:
            j += 1
        for k in range(i, j):
            o, l, t = off_rows[k], onl_rows[k], tim_rows[k]
            exp_cell = rf"\multirow{{{j-i}}}{{*}}{{{o['experiment']}}}" if k == i else ""
            # AC-OPF is nonconvex and generally intractable to solve exactly,
            # so v(theta) is NEVER genuinely available there (only relaxation
            # values of varying tightness stand in for it).
            vtrue_cell = "--" if o["experiment"] == "AC-OPF" else num(o["mean_vtrue"])
            tp = "--" if not np.isfinite(l["TP"]) else f"{int(round(l['TP']))}"
            fp = "--" if not np.isfinite(l["FP"]) else f"{int(round(l['FP']))}"
            tn = "--" if not np.isfinite(l["TN"]) else f"{int(round(l['TN']))}"
            fn = "--" if not np.isfinite(l["FN"]) else f"{int(round(l['FN']))}"
            relax_cell = sec(t["relax_solve_s"]) + (r"\textsuperscript{$\dagger$}" if t.get("dagger") else "")
            cells = [
                num(o["mean_vr"]), vtrue_cell, num(o["mean_f"]), num(o["mae"]), num(o["q_offset"]),
                num(l["delta"]), tp, fp, tn, fn,
                relax_cell, sec(t["local_solve_s"]),
            ]
            out.append("  " + " & ".join([exp_cell, o["relaxation"], _texcase(o["case"])] + cells) + r" \\")
        out.append(r"\midrule")
        i = j
    if out and out[-1] == r"\midrule":
        out.pop()
    return "\n".join(out)


def _splice_case2869_sdp(off_rows, onl_rows, tim_rows):
    """case2869pegase's chordal SDP has no offline/online row (no
    certification was ever computed for it -- see the paper's dagger
    footnote), but DOES have a timing-only measurement. Splice a synthetic
    row into all three lists (blank offline/online) right after the AC-OPF
    group's last row, so it renders inside AC-OPF's multirow block rather
    than appended after Knapsack."""
    acopf_idx = [i for i, r in enumerate(off_rows) if r["experiment"] == "AC-OPF"]
    insert_at = acopf_idx[-1] + 1
    blank = dict(experiment="AC-OPF", relaxation="SDP", case="case2869pegase",
                mean_vr=None, mean_vtrue=None, mean_f=None, mae=None, q_offset=None)
    blank_onl = dict(experiment="AC-OPF", relaxation="SDP", case="case2869pegase",
                     delta=None, TP=np.nan, FP=np.nan, TN=np.nan, FN=np.nan)
    # from results/acopf-cert/acopf_chordal_sdp_case2869pegase/solve_times.json:
    # only 1 of 5 attempted instances converged -- a single sample, flagged
    # with a dagger in the caption rather than presented as a stable mean.
    tim_row = dict(experiment="AC-OPF", relaxation="SDP", case="case2869pegase",
                   relax_solve_s=40.327943418, local_solve_s=0.4667142857142857,
                   dagger=True)
    off_rows.insert(insert_at, blank)
    onl_rows.insert(insert_at, blank_onl)
    tim_rows.insert(insert_at, tim_row)


def combined_table(off_rows, onl_rows, tim_rows):
    return rf"""\begin{{landscape}}
\begin{{table}}[p]
\centering\footnotesize
\caption{{All results combined, at the fixed operating point $\delta = 3\%\cdot\mathrm{{std}}(v(\bm\theta))$
and $\alpha = 1\%$ (offset $q_{{99}}$), averaged over the four folds. ${{}}^\ast$AC-OPF is nonconvex and
generally intractable to solve exactly, so genuine $v(\bm\theta)$ is never available there (only
relaxation values of varying tightness); shown as --. Timing: both columns of a given row are
measured identically (like-for-like); the local solver does not depend on the relaxation it is
compared against, so its solve time is measured once per experiment (per AC-OPF case; once for
inverse kinematics) and reused across relaxation rows. For AC-OPF, whose local solver (IPOPT) is
invoked through a file-based modelling interface, we report \emph{{solver-internal}} time on both
sides (parsed from IPOPT's own timing log for the local column) so the comparison is not dominated
by Python model construction or process-launch overhead. Mean over $50$ random instances per case
(smallest six AC-OPF cases) or $3$--$8$ for \texttt{{case1354pegase}}/\texttt{{case2869pegase}}'s SOCP
and \texttt{{case1354pegase}}'s chordal SDP; non-converged instances excluded from both timing
columns. \textsuperscript{{$\dagger$}}For \texttt{{case2869pegase}}'s chordal SDP, only $1$ of $5$
attempted instances converged (each costing $\sim\!70$s regardless of outcome); we report that
single solve time rather than assert a stable average, and omit its offline/online columns entirely
since no certification was computed for it, consistent with treating this relaxation as effectively
intractable at scale (\S\ref{{subsec:acopf-experiment}}).}}
\label{{tab:combined}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lll rrr rr r rrrr rr}}
\toprule
& & & \multicolumn{{5}}{{c}}{{Offline}} & \multicolumn{{5}}{{c}}{{Online}} & \multicolumn{{2}}{{c}}{{Timing}} \\
\cmidrule(lr){{4-8}}\cmidrule(lr){{9-13}}\cmidrule(lr){{14-15}}
\textbf{{Experiment}} & \textbf{{Relaxation}} & \textbf{{Case}} &
$\bar v_r(\bm\theta)$ & $\bar v(\bm\theta)^\ast$ & $\bar f(\hat{{\mathbf{{x}}}};\bm\theta)$ & MAE & $q_{{99}}$ &
$\delta$ & TP & FP & TN & FN &
Relax.\ Solve & Local Solve \\
\midrule
{_grouped_body(off_rows, onl_rows, tim_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
\end{{landscape}}"""


def main():
    off_rows = _load("offline")
    onl_rows = _load("online")
    tim_rows = _load("timing")
    assert len(off_rows) == len(onl_rows) == len(tim_rows), "row-set mismatch across tables"
    _splice_case2869_sdp(off_rows, onl_rows, tim_rows)

    tex = combined_table(off_rows, onl_rows, tim_rows)
    out = RESULTS / "tables_generated.tex"
    out.write_text(tex + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
