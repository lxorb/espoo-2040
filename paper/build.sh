#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 figs/collect.py "$@"
uv run --project .. --with matplotlib python figs/make.py > /dev/null
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdfinfo main.pdf | grep Pages
