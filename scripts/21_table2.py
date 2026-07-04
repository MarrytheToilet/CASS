"""Generate Table 2 (ablation matrix) as ONE full-width grid:
one row per axis; first data column = frozen default; remaining columns =
alternatives. Cell = small setting label stacked over accuracy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results" / "llama31-8b"
d = pd.read_csv(out / "e4_ablations.csv")
rr = pd.read_csv(out / "review_response.csv")
imp = pd.read_csv(out / "improvements.csv")
e1 = pd.read_csv(out / "e1_loto.csv")
pre = pd.read_csv(out / "v3_pregate" / "e1_loto.csv")


def ax_acc(axis, v):
    sub = d[(d.axis == axis) & (d.value == str(v))]
    return sub["acc"].mean() if len(sub) else np.nan


cass32 = e1[(e1["mode"] == "cass") & (e1["k"] == 4) &
            (e1["seed"] < 3)].groupby("task")["acc"].mean().mean()
ungated = pre[(pre["mode"] == "cass") & (pre["k"] == 4) &
              (pre["seed"] < 3)].groupby("task")["acc"].mean().mean()

# rows: (axis label, default (lab, acc), [(lab, acc), ...alternatives])
ROWS = [
    ("Shared rank $r_0$", ("$r_0{=}1$", ax_acc("r0", 1)),
     [("$r_0{=}0$", ax_acc("r0", 0)), ("$r_0{=}2$", ax_acc("r0", 2)),
      ("$r_0{=}4$", ax_acc("r0", 4)), ("$r_0{=}8$", ax_acc("r0", 8)),
      ("$r_0{=}16$", ax_acc("r0", 16)),
      ("random dir.", rr.query("exp=='control'")["acc"].mean())]),
    ("Injection layers", ("$\\{12,16\\}$", ax_acc("layers", "12+16")),
     [("$\\{12\\}$", ax_acc("layers", "12")),
      ("$\\{16\\}$", ax_acc("layers", "16")),
      ("$\\{12,14,16\\}$", ax_acc("layers", "12+14+16"))]),
    ("Direction $\\Delta$", ("hybrid ($z$)", ax_acc("delta", "hybrid")),
     [("$z$-only add.", ax_acc("delta", "zonly")),
      ("reconstruction", ax_acc("delta", "recon")),
      ("anchor blend", ax_acc("delta", "blend")),
      ("projection only", ax_acc("delta", "projection"))]),
    ("Solver", ("group LASSO", ax_acc("solver", "group_lasso")),
     [("block OMP", ax_acc("solver", "omp")),
      ("simplex", ax_acc("solver", "simplex")),
      ("least squares", ax_acc("solver", "ls"))]),
    ("Examples $k$", ("$k{=}4$", ax_acc("k", 4)),
     [("$k{=}1$", ax_acc("k", 1)), ("$k{=}2$", ax_acc("k", 2)),
      ("$k{=}8$", ax_acc("k", 8))]),
    ("Support cap", ("$s_{\\max}{=}5$", ax_acc("smax", 5)),
     [("$s_{\\max}{=}3$", ax_acc("smax", 3)),
      ("$s_{\\max}{=}8$", ax_acc("smax", 8)),
      ("$s_{\\max}{=}31$", ax_acc("smax", 31))]),
    ("Samples per skill", ("$n{=}100$", ax_acc("n", 100)),
     [("$n{=}25$", ax_acc("n", 25)), ("$n{=}50$", ax_acc("n", 50))]),
    ("Trust gate$^\\dagger$", ("global gate", cass32),
     [("ungated", ungated),
      ("per-skill signed", imp[imp.cond == "signed"]["acc"].mean())]),
    ("Injection schedule$^\\dagger$", ("every step", cass32),
     [("prefill only", imp[imp.cond == "prefill_only"]["acc"].mean())]),
]

MAXALT = max(len(alts) for _, _, alts in ROWS)


NOTES = {
    "Shared rank $r_0$": "",
    "Injection layers": "both depths necessary",
    "Direction $\\Delta$": "direction must come from $z$",
    "Solver": "insensitive: sparsity, not the solver, matters",
    "Examples $k$": "saturates at $k{=}4$",
    "Support cap": "flat for 3--8",
    "Samples per skill": "25 samples suffice",
    "Trust gate$^\\dagger$": "gate converts failures (\\S6.6)",
    "Injection schedule$^\\dagger$": "decode-step steering matters",
}


def cell(lab, acc, bold=False):
    a = "?" if acc != acc else f"{acc:.2f}"
    if bold:
        return f"{{\\scriptsize {lab}}}~\\textbf{{{a}}}"
    return f"{{\\scriptsize\\color{{gray}} {lab}}}~{a}"


lines = []
for axis, (dlab, dacc), alts in ROWS:
    cells = [cell(dlab, dacc, bold=True)]
    cells += [cell(l, a) for l, a in alts]
    free = MAXALT - len(alts)
    row = f"{axis} & " + " & ".join(cells)
    note = NOTES.get(axis, "")
    if free > 0 and note:
        row += (f" & \\multicolumn{{{free}}}{{r@{{}}}}"
                f"{{{{\\scriptsize\\itshape\\color{{gray}} {note}}}}}")
    elif free > 0:
        row += " &" * free
    lines.append(row + "\\\\[3.5pt]")
body = "\n".join(lines)

tex = (r"""\begin{table*}[t]
\caption{Ablation matrix (mean accuracy; 8 representative tasks, $k{=}4$,
3 seeds, unless marked $^\dagger$ = all 32 LOTO tasks). One row per design
axis; the first column (bold) is the frozen default configuration, and each
alternative varies that axis with all others held at their defaults.
Per-setting residuals and support sizes are in the appendix.}
\label{tab:ablation}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l""" +
       "c" * (1 + MAXALT) + r"""@{}}
\toprule
Axis & default & \multicolumn{""" + str(MAXALT) +
       r"""}{c}{alternatives}\\
\cmidrule(lr){2-2}\cmidrule(lr){3-""" + str(2 + MAXALT) + r"""}
""" + body + r"""
\bottomrule
\end{tabular*}
\end{table*}""")
open(ROOT / "paper" / "table2.tex", "w").write(tex)
missing = tex.count("?")
print(f"table2.tex written ({missing} cells pending E4 rerun)")
