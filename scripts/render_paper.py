"""Render the manuscript's deliberately small Markdown subset to LaTeX."""

from __future__ import annotations

import re
from pathlib import Path


def escape(text):
    symbols = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "±": r"\(\pm\)",
        "×": r"\(\times\)",
        "“": "``",
        "”": "''",
        "–": "--",
        "—": "---",
        "−": "-",
        "’": "'",
    }
    return "".join(symbols.get(c, c) for c in text)


def inline(text):
    tokens = []

    def protect(value):
        tokens.append(value)
        return f"ZZPROTECTED{len(tokens) - 1}ZZ"

    text = re.sub(r"\$([^$\n]+)\$", lambda m: protect("$" + m[1] + "$"), text)
    text = re.sub(r"`([^`]+)`", lambda m: protect(r"\texttt{" + escape(m[1]) + "}"), text)
    text = re.sub(
        r"https?://[^\s)]+",
        lambda m: protect(r"\url{" + m[0].rstrip(".") + "}") + ("." if m[0].endswith(".") else ""),
        text,
    )
    text = re.sub(
        r"\b(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", lambda m: protect(r"\path{" + m[0] + "}"), text
    )
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    for i, value in enumerate(tokens):
        text = text.replace(f"ZZPROTECTED{i}ZZ", value)
    return text


def render(markdown: str, root: Path) -> str:
    lines = markdown.splitlines()
    if not lines[0].startswith("# "):
        raise ValueError("Missing manuscript title")
    author_end = next(i for i in range(2, len(lines)) if not lines[i].strip())
    author = r"\\".join(inline(line.strip()) for line in lines[2:author_end])
    body, paragraph = [], []
    abstract = False

    def flush():
        if paragraph:
            body.append(inline(" ".join(paragraph)) + "\n")
            paragraph.clear()

    i = author_end + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            # Keep a table's description with its unbreakable tabular block.
            if not paragraph or not re.match(r"^Table \d", paragraph[0]):
                flush()
        elif line.startswith("<!--"):
            pass
        elif line == "$$":
            flush()
            equation = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                equation.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError("Unclosed equation")
            body.append("\\[\n" + "\n".join(equation) + "\n\\]\n")
        elif line.startswith(("~~~", "```")):
            flush()
            delimiter, code = line[:3], []
            i += 1
            while i < len(lines) and not lines[i].startswith(delimiter):
                code.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError("Unclosed code block")
            body.append(
                "\\begin{Verbatim}[fontsize=\\small]\n" + "\n".join(code) + "\n\\end{Verbatim}\n"
            )
        elif line.startswith("##"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            heading = line[level:].strip()
            if abstract:
                body.append("\\end{abstract}\n")
                abstract = False
            if heading == "Abstract":
                body.append("\\begin{abstract}\n")
                abstract = True
            else:
                heading = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", heading)
                command = "section" if level == 2 else "subsection"
                if heading.startswith("Appendix") or heading == "References":
                    command += "*"
                body.append("\\" + command + "{" + inline(heading) + "}\n")
        elif line.startswith("|"):
            caption = None
            if paragraph and re.match(r"^Table \d", paragraph[0]):
                caption = inline(" ".join(paragraph))
                paragraph.clear()
            else:
                flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            i -= 1
            n = len(rows[0])
            if any(len(row) != n for row in rows):
                raise ValueError("Ragged table")
            cols = r">{\raggedright\arraybackslash}X" * n
            table = [
                r"\begin{center}",
                r"\small",
                r"\setlength{\tabcolsep}{3pt}",
                r"\renewcommand{\arraystretch}{1.15}",
                r"\begin{tabularx}{\linewidth}{" + cols + "}",
                r"\toprule",
                " & ".join(r"\textbf{" + inline(c) + "}" for c in rows[0]) + r" \\",
                r"\midrule",
            ]
            table += [" & ".join(inline(c) for c in row) + r" \\" for row in rows[1:]]
            table += [r"\bottomrule", r"\end{tabularx}", "\\end{center}\n"]
            if caption:
                table = (
                    [r"\noindent\begin{minipage}{\linewidth}", caption + r"\par"]
                    + table
                    + [r"\end{minipage}\par"]
                )
            body.append("\n".join(table))
        elif line.startswith("!["):
            flush()
            match = re.fullmatch(r"!\[(.*)\]\(([^)]+)\)", line)
            if not match:
                raise ValueError("Malformed figure")
            source = root / match[2]
            if not source.exists():
                raise FileNotFoundError(source)
            if source.with_suffix(".pdf").exists():
                source = source.with_suffix(".pdf")
            relative = Path("..") / source.relative_to(root)
            body.append(
                "\\begin{figure}[!htbp]\n\\centering\n"
                "\\includegraphics[width=\\linewidth]{" + relative.as_posix() + "}\n"
                "\\caption{" + inline(match[1]) + "}\n\\end{figure}\n"
            )
        elif re.match(r"^\d+\.\s", line):
            flush()
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s", "", lines[i].strip()))
                i += 1
            i -= 1
            body.append(
                "\\begin{enumerate}[itemsep=2pt]\n"
                + "\n".join("\\item " + inline(x) for x in items)
                + "\n\\end{enumerate}\n"
            )
        else:
            paragraph.append(line)
        i += 1
    flush()
    if abstract:
        body.append(r"\end{abstract}")
    preamble = r"""\documentclass[11pt,a4paper]{article}
% Generated from PAPER.md. Edit the Markdown source.
\usepackage[margin=25mm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,array}
\usepackage{xcolor,hyperref,fancyvrb,enumitem}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{4pt}
\title{TITLE}
\author{AUTHOR}
\date{}
\begin{document}
\maketitle
""".replace("TITLE", inline(lines[0][2:])).replace("AUTHOR", author)
    return preamble + "\n".join(body) + "\n\\clearpage\n\\end{document}\n"
