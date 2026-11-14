#!/bin/bash
cd "$(dirname "$0")"
sed 's/^\\textonlyfalse/\\textonlytrue/' main.tex > textonly.tex
pdflatex -interaction=nonstopmode textonly.tex > /dev/null; pdflatex -interaction=nonstopmode textonly.tex > /dev/null
pdfinfo textonly.pdf | grep Pages
rm -f textonly.aux textonly.log textonly.out textonly.tex
