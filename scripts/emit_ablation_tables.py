"""Emit the three NN ablation tables (depth, training budget, width) as LaTeX,
from the merged AC-OPF ablation grid. Writes results/ablation_tables.tex.

Source: results/acopf/ablation_summary.csv, produced by the SAVIO job array
(scripts/submit_ablation_grid.sh) and merged with scripts/merge_ablation_parts.py.

All three tables are cut from ONE grid run with a SINGLE consistent protocol --
the single-phase cosine schedule of the main pipeline (train_generic.py), n=20k,
4-fold CV, SOCP relaxation -- varying one factor at a time around the reference
configuration (6 hidden layers x 256 units, 1000 epochs):

    depth  : 1..6                        @ width 256, 1000 epochs
    epochs : 100,200,500,1000,1500,2000  @ 6x256
    width  : 64,128,256,512              @ depth 6, 1000 epochs

Earlier runs used a two-phase schedule and are filtered out (single_phase == 1)
so that a single protocol backs every reported number.

Metric: the PRIMARY quantity is the upper-tail overprediction error that the
certification framework is actually sensitive to -- q95_overpred_pct, the 95th
percentile of v_hat's OVERprediction of the relaxation value v_r, as a percent
of v_r (this is the same quantity q_alpha calibrates against, just uncalibrated
and reported as a fold-pooled percentage rather than an absolute conformal
offset). Mean absolute percent error (MAPE) is a poor proxy for this: a network
can have low average error while still overpredicting badly on the tail that
determines whether f <= v_hat + q_alpha - delta ever certifies. We report MAPE
in parentheses for context, but architecture/budget choices should be judged on
the tail column.

Usage:  python scripts/emit_ablation_tables.py
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "acopf"
OUT = ROOT / "results" / "ablation_tables.tex"

CASES = ["case9", "case118", "case1354pegase"]
CLABEL = {"case9": r"\texttt{case9}", "case118": r"\texttt{case118}",
          "case1354pegase": r"\texttt{case1354}"}
# REF_* is the center point the ORIGINAL grid varied one factor around
# (6x256, 1000 epochs) -- kept for the width/depth-at-256/epoch tables below,
# which are the actual runs that exist. DEFAULT_* is the adopted final
# recipe (4x64, 1500 epochs), picked from the width table's q95 result --
# used for the depth-at-width-64 table, which is the ablation that actually
# matters for judging the chosen depth.
REF_DEPTH, REF_WIDTH, REF_EPOCHS = 6, 256, 1000
DEFAULT_WIDTH, DEFAULT_EPOCHS = 64, 1000  # depth ablation at width=64 only exists at 1000 epochs
TIME_CASE = "case1354pegase"          # representative train-time column


def pct(x):
    return "--" if pd.isna(x) else f"{100*x:.2f}"


def cell(row):
    """q95 upper-tail overprediction (primary), MAPE in parentheses (context)."""
    if row is None:
        return "--"
    q95 = pct(row["q95_overpred_pct"])
    mape = pct(row["ape_relax_mean"])
    return rf"{q95} ({mape})"


def load():
    d = pd.read_csv(RESULTS / "ablation_summary.csv")
    d["epochs"] = d["pretrain_epochs"].fillna(0) + d["finetune_epochs"].fillna(0)
    if "single_phase" in d.columns:
        sp = d[d["single_phase"] == 1]
        if len(sp):
            d = sp
        else:
            print("  WARNING: no single-phase rows; falling back to all rows "
                  "(protocols may be mixed)", file=sys.stderr)
    return d


def _table(d, axis, values, fixed, label, caption, head):
    """One ablation table: `axis` varies over `values`, everything else `fixed`."""
    rows = []
    for v in values:
        sub = d[d[axis] == v]
        for k, fv in fixed.items():
            sub = sub[sub[k] == fv]
        cells = []
        for c in CASES:
            r = sub[sub["case"] == c]
            cells.append(cell(r.iloc[0]) if len(r) else "--")
        tr = sub[sub["case"] == TIME_CASE]
        t = f"{tr['train_time_s'].iloc[0]:.0f}" if len(tr) else "--"
        rows.append(f"  ${int(v)}$ & " + " & ".join(cells) + f" & {t} \\\\")
    body = "\n".join(rows)
    heads = " & ".join(CLABEL[c] for c in CASES)
    return rf"""\begin{{table}}[h]
\centering\footnotesize
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{c ccc r}}
\toprule
\textbf{{{head}}} & {heads} & \textbf{{Train (s)}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main():
    d = load()
    tables = []

    tables.append(_table(
        d, "depth", sorted(d[(d["width"] == REF_WIDTH) & (d["epochs"] == REF_EPOCHS)]["depth"].unique()),
        {"width": REF_WIDTH, "epochs": REF_EPOCHS},
        "tab:ablation-depth",
        rf"Network-\emph{{depth}} ablation on AC-OPF (SOCP relaxation), at fixed width "
        rf"${REF_WIDTH}$ and ${REF_EPOCHS}$ epochs: $4$-fold cross-validated $95$th-percentile "
        rf"overprediction of $\hat{{v}}$ above $v_r$ (\%), the certification-relevant tail "
        rf"quantity, with mean absolute percent error (MAPE) in parentheses for context. "
        rf"Train time is for \texttt{{case1354}} (seconds, all four folds).",
        "Hidden layers"))

    tables.append(_table(
        d, "epochs", sorted(d[(d["depth"] == REF_DEPTH) & (d["width"] == REF_WIDTH)]["epochs"].unique()),
        {"depth": REF_DEPTH, "width": REF_WIDTH},
        "tab:ablation-epochs",
        rf"Training-\emph{{budget}} ablation on AC-OPF (SOCP relaxation, "
        rf"${REF_DEPTH}\times{REF_WIDTH}$ network): $4$-fold cross-validated $95$th-percentile "
        rf"overprediction of $\hat{{v}}$ above $v_r$ (\%), the certification-relevant tail "
        rf"quantity, with MAPE in parentheses for context, versus the number of training "
        rf"epochs. Each budget is a separate run with the cosine schedule annealed over that "
        rf"budget. Train time is for \texttt{{case1354}} (seconds, all four folds).",
        "Epochs"))

    tables.append(_table(
        d, "width", sorted(d[(d["depth"] == REF_DEPTH) & (d["epochs"] == REF_EPOCHS)]["width"].unique()),
        {"depth": REF_DEPTH, "epochs": REF_EPOCHS},
        "tab:ablation-width",
        rf"Layer-\emph{{width}} ablation on AC-OPF (SOCP relaxation), at fixed depth "
        rf"${REF_DEPTH}$ and ${REF_EPOCHS}$ epochs: $4$-fold cross-validated $95$th-percentile "
        rf"overprediction of $\hat{{v}}$ above $v_r$ (\%), the certification-relevant tail "
        rf"quantity, with MAPE in parentheses for context. Width $64$ wins on every case on "
        rf"this tail metric, which is why it -- not the deeper $256$-unit default this grid "
        rf"was centered on -- is the width adopted throughout the paper. Train time is for "
        rf"\texttt{{case1354}} (seconds, all four folds).",
        "Layer width"))

    tables.append(_table(
        d, "depth", sorted(d[(d["width"] == DEFAULT_WIDTH) & (d["epochs"] == DEFAULT_EPOCHS)]["depth"].unique()),
        {"width": DEFAULT_WIDTH, "epochs": DEFAULT_EPOCHS},
        "tab:ablation-depth-w64",
        rf"Network-\emph{{depth}} ablation at the \emph{{adopted}} width $64$ (SOCP relaxation, "
        rf"${DEFAULT_EPOCHS}$ epochs): $4$-fold cross-validated $95$th-percentile overprediction "
        rf"of $\hat{{v}}$ above $v_r$ (\%), MAPE in parentheses. Depth $5$--$6$ edge out depth $4$ "
        rf"on this tail metric on \texttt{{case118}} ($1.74$/$1.78$ vs.\ $2.07$); we adopt depth "
        rf"$4$ regardless as a practical plateau point with roughly $30$--$40\%$ fewer parameters "
        rf"and comparable training cost, not as the strict per-case optimum. Train time is for "
        rf"\texttt{{case1354}} (seconds, all four folds).",
        "Hidden layers"))

    OUT.write_text("\n\n".join(tables) + "\n")
    print(f"wrote {OUT}")
    for name, ax in (("depth", "depth"), ("epochs", "epochs"), ("width", "width")):
        print(f"  {name}: {sorted(d[ax].unique())}")


if __name__ == "__main__":
    main()
