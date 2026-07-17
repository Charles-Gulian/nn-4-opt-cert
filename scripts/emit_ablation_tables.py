"""Emit the two NN ablation tables (depth and training epochs) as LaTeX, from the
existing AC-OPF ablation runs. Writes results/ablation_tables.tex.

Metric: 4-fold cross-validated mean absolute percent error (MAPE) of v_hat against
the relaxation value v_r, in percent. Depth table fixes width 256, 1000 epochs;
epoch table fixes the 6x256 architecture.

Usage:  python scripts/emit_ablation_tables.py
"""
import pathlib
import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "acopf"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "ablation_tables.tex"

CASES = ["case9", "case118", "case1354pegase"]
CLABEL = {"case9": r"\texttt{case9}", "case118": r"\texttt{case118}",
          "case1354pegase": r"\texttt{case1354}"}


def pct(x):
    return "--" if pd.isna(x) else f"{100*x:.2f}"


def depth_table():
    d = pd.read_csv(RESULTS / "ablation_summary.csv")
    d = d[d["width"] == 256]
    depths = sorted(d["depth"].unique())
    rows = []
    for dep in depths:
        cells = []
        for c in CASES:
            r = d[(d["depth"] == dep) & (d["case"] == c)]
            cells.append(pct(r["ape_relax_mean"].iloc[0]) if len(r) else "--")
        # representative train time: the largest case where available
        tr = d[(d["depth"] == dep) & (d["case"] == "case1354pegase")]
        t = f"{tr['train_time_s'].iloc[0]:.0f}" if len(tr) else "--"
        rows.append(f"  ${int(dep)}$ & " + " & ".join(cells) + f" & {t} \\\\")
    body = "\n".join(rows)
    heads = " & ".join(CLABEL[c] for c in CASES)
    return rf"""\begin{{table}}[h]
\centering\footnotesize
\caption{{Network-\emph{{depth}} ablation on AC-OPF (SOCP relaxation): $4$-fold
cross-validated MAPE (\%) of $\hat{{v}}$ against $v_r$, at fixed width $256$ and $1000$
epochs. Error falls steeply with depth and plateaus by $4$--$6$ hidden layers, which
motivates our choice of six. Train time is for \texttt{{case1354}} (seconds, all four folds).}}
\label{{tab:ablation-depth}}
\begin{{tabular}}{{c ccc r}}
\toprule
\textbf{{Hidden layers}} & {heads} & \textbf{{Train (s)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def epoch_table():
    d = pd.read_csv(RESULTS / "ablation_epochs.csv")
    d["epochs"] = d["pretrain_epochs"] + d["finetune_epochs"]
    # prefer single-phase where both exist (matches the recipe); one row per epoch budget
    d = d.sort_values(["epochs", "single_phase"])
    rows = []
    for e in sorted(d["epochs"].unique()):
        sub = d[d["epochs"] == e]
        sp = sub[sub["single_phase"] == 1]
        r = sp if len(sp) else sub
        c118 = r[r["case"] == "case118"]
        c1354 = r[r["case"] == "case1354pegase"]
        v118 = pct(c118["ape_relax_mean"].iloc[0]) if len(c118) else "--"
        v1354 = pct(c1354["ape_relax_mean"].iloc[0]) if len(c1354) else "--"
        t = f"{c1354['train_time_s'].iloc[0]:.0f}" if len(c1354) else "--"
        rows.append(f"  ${int(e)}$ & {v118} & {v1354} & {t} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[h]
\centering\footnotesize
\caption{{Training-\emph{{budget}} ablation on AC-OPF (SOCP relaxation, $6\times256$
network): $4$-fold cross-validated MAPE (\%) of $\hat{{v}}$ against $v_r$ versus the number
of training epochs. Error decreases monotonically and largely plateaus by $1000$ epochs,
our chosen budget. Train time is for \texttt{{case1354}} (seconds, all four folds).}}
\label{{tab:ablation-epochs}}
\begin{{tabular}}{{c cc r}}
\toprule
\textbf{{Epochs}} & \texttt{{case118}} & \texttt{{case1354}} & \textbf{{Train (s)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main():
    OUT.write_text(depth_table() + "\n\n" + epoch_table() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
