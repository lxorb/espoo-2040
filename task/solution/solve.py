import json
import os
import sys
import urllib.request
from pathlib import Path

URL = "http://mcp-server:8000/mcp"


def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["result"]


def call(tool, **args):
    r = rpc("tools/call", {"name": tool, "arguments": args})
    text = r["content"][0]["text"]
    if r.get("isError"):
        raise RuntimeError(text)
    return json.loads(text)


def learned(st, projects, tried):
    proj = [dict(id=p[0], kind=p[2], cost=p[3], years=p[4], opex=p[5]) for p in projects]
    h = min(st["rules"]["forecast_years"], st["years_left"] + st["runout_left"])
    fc = lambda a: call("forecast", plan={"0": a}, years=h)["total_loss_kh"]
    return POLICY(st, proj, [p["id"] in st["startable"] for p in proj], fc, st["forecasts_left"], st["rules"]["years"])


def greedy(st, projects, tried):
    menu = {p[0]: p for p in projects}
    if not set(st["startable"]) - tried:
        tried.clear()
    cands = [k for k in st["startable"] if k not in tried][:st["forecasts_left"] - 1]
    h = min(st["rules"]["forecast_years"], st["years_left"] + st["runout_left"])
    base = call("forecast", plan={"0": []}, years=h)["total_loss_kh"]
    best, pick = 0, []
    for k in cands:
        tried.add(k)
        f = call("forecast", plan={"0": [{"build": k}]}, years=h)
        if f["feasible"] and (g := (base - f["total_loss_kh"]) / max(menu[k][3], 0.1)) > best:
            best, pick = g, [{"build": k}]
    return pick


sys.path.insert(0, str(Path(__file__).parent))
POLICY = __import__("policy").Policy() if Path(__file__).with_name("policy.npz").exists() else None
rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "oracle", "version": "0"}})
stop = 3 if os.environ.get("MODE") == "early" else None
tried = set()
while True:
    st = call("get_state")
    if not st["years_left"] or stop == 0:
        break
    projects = call("list_projects")["rows"]
    print(call("commit_year", actions=(learned if POLICY else greedy)(st, projects, tried)))
    stop = stop and stop - 1
