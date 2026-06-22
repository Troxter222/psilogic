"""Post-hoc analysis: turn result CSVs into publication-ready LaTeX tables.

Usage::

    python -m fairbench.analysis --output-dir results/full --metric val_acc

Produces a LaTeX ``table`` (booktabs) of Mean+/-Std per optimizer per arena,
bolding the best optimizer and annotating PsiLogic cells with significance
stars from the Welch t-test (``*`` p<0.05, ``**`` p<0.01, ``***`` p<0.001).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def build_latex_table(
    output_dir: str, metric: str, lower_is_better: bool = True, reference: str = "psilogic"
) -> str:
    """Build a LaTeX table string for ``final_<metric>`` across arenas/optimizers."""
    agg = _read_csv(os.path.join(output_dir, "aggregate.csv"))
    sig = _read_csv(os.path.join(output_dir, "significance.csv"))
    if not agg:
        return "% No aggregate.csv found -- run the benchmark first.\n"

    # aggregate.csv stores aggregated columns under bare metric names.
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    # cell[arena][optimizer] = (mean, std)
    cell: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    optimizers: list[str] = []
    for row in agg:
        if mean_key not in row or row.get(mean_key, "") == "":
            continue
        arena, opt = row["arena"], row["optimizer"]
        try:
            cell[arena][opt] = (float(row[mean_key]), float(row.get(std_key, "nan")))
        except ValueError:
            continue
        if opt not in optimizers:
            optimizers.append(opt)

    # pval[arena][baseline] for the metric (reference vs baseline)
    pval: dict[str, dict[str, float]] = defaultdict(dict)
    for row in sig:
        if row.get("metric") != f"final_{metric}" and row.get("metric") != metric:
            continue
        try:
            pval[row["arena"]][row["baseline"]] = float(row["p_value"])
        except (KeyError, ValueError):
            pass

    arenas = sorted(cell)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Final {metric.replace('_', ' ')} (mean $\\pm$ std over seeds). "
        f"Best per arena in \\textbf{{bold}}; stars mark {reference} significance "
        "(\\textsuperscript{*}$p<0.05$, \\textsuperscript{**}$p<0.01$, \\textsuperscript{***}$p<0.001$).}",
        "\\label{tab:" + metric + "}",
        "\\begin{tabular}{l" + "c" * len(optimizers) + "}",
        "\\toprule",
        "Arena & " + " & ".join(o.capitalize() for o in optimizers) + " \\\\",
        "\\midrule",
    ]

    for arena in arenas:
        vals = {o: cell[arena][o][0] for o in optimizers if o in cell[arena]}
        if vals:
            best_opt = (min if lower_is_better else max)(vals, key=vals.get)
        else:
            best_opt = None
        cells = []
        for opt in optimizers:
            if opt not in cell[arena]:
                cells.append("--")
                continue
            mean, std = cell[arena][opt]
            txt = f"{mean:.3f} $\\pm$ {std:.3f}"
            if opt == reference:
                # Significance annotation lives with the reference cell only if
                # we compare it against the best baseline; here we star per-arena
                # using its smallest p-value against any baseline.
                ps = [
                    pval[arena].get(b)
                    for b in optimizers
                    if b != reference and pval[arena].get(b) is not None
                ]
                if ps:
                    txt += f"\\textsuperscript{{{_stars(min(ps))}}}"
            if opt == best_opt:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(f"{arena} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate a LaTeX results table from benchmark CSVs.")
    p.add_argument(
        "--output-dir", default="./results", help="Directory containing the result CSVs."
    )
    p.add_argument(
        "--metric", default="val_loss", help="Metric column (without the 'final_' prefix)."
    )
    p.add_argument(
        "--higher-better",
        action="store_true",
        help="Treat higher metric as better (e.g. accuracy).",
    )
    p.add_argument("--out", default=None, help="Write the LaTeX to this file instead of stdout.")
    args = p.parse_args(argv)

    latex = build_latex_table(args.output_dir, args.metric, lower_is_better=not args.higher_better)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(latex)
        print(f"Wrote {args.out}")
    else:
        print(latex)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
