#!/bin/bash
set -e
cd "$(dirname "$0")"
[ -d ../runs ] && [ -d ../jobs ] && python3 figs/collect.py "$@"
uv run --no-project --with numpy --with scipy --with matplotlib python figs/make.py
latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdfinfo build/main.pdf | grep Pages
