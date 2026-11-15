#!/bin/bash
cd "$(dirname "$0")"
n=$(pdfinfo main.pdf | awk '/Pages/{print $2}'); last=$n
for p in $(seq 1 $n); do
  r=$(pdftotext -f $p -l $p -layout main.pdf - | grep . | grep -n -m1 -E '^\s*References\s*$' | cut -d: -f1)
  [ -n "$r" ] && { [ "$r" -le 6 ] && last=$((p - 1)) || last=$p; break; }
done
rm -rf ../viz/assets/paper && mkdir -p ../viz/assets/paper
pdftoppm -r 150 -f 1 -l $last -jpeg -jpegopt quality=80 main.pdf ../viz/assets/paper/p
k=0; for f in ../viz/assets/paper/p-*.jpg; do k=$((k + 1)); mv "$f" ../viz/assets/paper/$k.jpg; done
echo "{\"n\": $k}" > ../viz/assets/paper/pages.json
cat ../viz/assets/paper/pages.json
