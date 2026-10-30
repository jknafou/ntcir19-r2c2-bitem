#!/usr/bin/env bash
# Build one figure: <name>.tex -> <name>.pdf -> <name>.svg
#
# pipeline.tex is drawn by hand. confidence-ac2.tex and composite-shapes.tex are written by
# analysis/confidence_figure.py and analysis/composite_shapes.py.
#
# Usage:
#   bash docs/figures/build.sh pipeline
#
# Captions are off by default, since the README carries that text in its own prose. CAPTION=1
# defines \withcaption, which both sources test, and builds the captioned figure for the paper:
#   CAPTION=1 bash docs/figures/build.sh pipeline
#
# Takes a figure basename (or a path to the .tex) and leaves the .pdf and .svg beside the
# source. LaTeX runs twice so that any \ref or node coordinate resolved through the .aux is
# stable. The SVG is produced by MuPDF with the text converted to outlines, so it carries no
# font dependency and renders the same everywhere.
#
# One-time setup. The SVG converter comes from MuPDF; the LaTeX side needs the standalone class,
# which a minimal TeX Live lacks, so it is installed into a private tree beside this script:
#   uv venv --python 3.12 docs/figures/.venv
#   uv pip install --python docs/figures/.venv/bin/python pymupdf
#   TEXMFHOME=docs/figures/.texmf tlmgr init-usertree
#   TEXMFHOME=docs/figures/.texmf tlmgr --usermode --ignore-warning \
#       --repository https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2020/tlnet-final \
#       install standalone preview adjustbox gincltex filemod
# On a machine with a full TeX Live the tlmgr steps are unnecessary.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_python="$here/.venv/bin/python"

# A private package tree, searched in addition to the system one.
if [ -d "$here/.texmf" ]; then
    export TEXMFHOME="$here/.texmf"
fi

if [ $# -ne 1 ]; then
    echo "usage: build.sh <figure-name|path/to/figure.tex>" >&2
    exit 2
fi

# Accept "pipeline", "pipeline.tex", or a path to a .tex anywhere.
arg="${1%.tex}"
if [ -f "$arg.tex" ]; then
    source_tex="$(cd "$(dirname "$arg.tex")" && pwd)/$(basename "$arg.tex")"
elif [ -f "$here/$(basename "$arg").tex" ]; then
    source_tex="$here/$(basename "$arg").tex"
else
    echo "build.sh: no such figure source: $arg.tex" >&2
    exit 2
fi

name="$(basename "$source_tex" .tex)"
pdf="$here/$name.pdf"
svg="$here/$name.svg"

if [ ! -x "$venv_python" ]; then
    echo "build.sh: missing $venv_python -- run the one-time setup at the top of this script" >&2
    exit 3
fi

echo "source: $source_tex"

# With CAPTION=1 the source is \input from the command line so \withcaption can be defined ahead
# of it; that route makes the job name "texput", hence the explicit -jobname on both branches.
tex_job="$source_tex"
if [ "${CAPTION:-0}" = 1 ]; then
    tex_job="\\def\\withcaption{}\\input{$source_tex}"
fi

for pass in 1 2; do
    lualatex -interaction=nonstopmode -halt-on-error -jobname="$name" \
             -output-directory="$here" "$tex_job" >"$here/$name.pass$pass.log" 2>&1 \
        || { echo "build.sh: lualatex pass $pass failed; see $here/$name.pass$pass.log" >&2; exit 1; }
done
rm -f "$here/$name.pass1.log" "$here/$name.pass2.log" "$here/$name.aux" "$here/$name.log"

[ -s "$pdf" ] || { echo "build.sh: lualatex produced no $pdf" >&2; exit 1; }
echo "pdf   : $pdf"

"$venv_python" "$here/pdf_to_svg.py" "$pdf" "$svg"
