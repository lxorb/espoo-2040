import json
import os
import urllib.request

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


rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "oracle", "version": "0"}})
stop = 3 if os.environ.get("MODE") == "early" else None
tried = set()
while True:
    st = call("get_state")
    if not st["years_left"] or stop == 0:
        break
    acts = [{"resurface": r[0]} for r in st["worst"]["rows"] if r[7] < 0.35] + [{"retime": r[0]} for r in st["congested"]["rows"][:10]]
    rows = [p for p in call("list_projects")["rows"] if p[7] == "open" and p[8]]
    if not {p[0] for p in rows} - tried:
        tried = set()
    cands = [p for p in rows if p[0] not in tried][:st["forecasts_left"] - 1]
    h = min(max([p[4] for p in cands], default=0) + 3, st["years_left"] + st["runout_left"])
    base = call("forecast", plan={"0": acts}, years=h)["total_delay_kh"]
    best, pick = 0, None
    for p in cands:
        tried.add(p[0])
        f = call("forecast", plan={"0": acts + [{"build": p[0]}]}, years=h)
        if f["feasible"] and (g := (base - f["total_delay_kh"]) / max(p[3], 0.1)) > best:
            best, pick = g, p[0]
    print(call("commit_year", actions=acts + ([{"build": pick}] if pick else [])))
    stop = stop and stop - 1
