import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

SITE, ROOT, PORT = "https://espoo.evinu.de/", Path("espoo-offline"), 8000
TYPES = {".js": "text/javascript", ".json": "application/json", ".webp": "image/webp", ".glb": "model/gltf-binary", ".woff2": "font/woff2", ".css": "text/css", ".html": "text/html"}


def get(path, md5=None):
    to = ROOT / path
    if md5 and to.exists() and hashlib.md5(to.read_bytes()).hexdigest() == md5:
        return
    to.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SITE + quote(path), headers={"User-Agent": "espoo-offline"})
    for _ in range(4):
        try:
            data = urllib.request.urlopen(req, timeout=60).read()
            if not md5 or hashlib.md5(data).hexdigest() == md5:
                return to.write_bytes(data)
        except OSError:
            pass
    print("missing", path)


if "serve" not in sys.argv:
    try:
        get("files.txt")
        rows = [r.split("  ./", 1) for r in (ROOT / "files.txt").read_text().splitlines()]
        with ThreadPoolExecutor(16) as pool:
            for k, _ in enumerate(pool.map(lambda r: get(r[1], r[0]), rows)):
                print(f"\r{k + 1} of {len(rows)} files", end="")
        print()
    except OSError:
        print("no network, serving the local copy")
SimpleHTTPRequestHandler.extensions_map.update(TYPES)
print(f"open http://localhost:{PORT}")
ThreadingHTTPServer(("127.0.0.1", PORT), partial(SimpleHTTPRequestHandler, directory=str(ROOT))).serve_forever()
