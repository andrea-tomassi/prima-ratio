#!/usr/bin/env python3
"""Push 2 parallel chat slots to a target context size and report VRAM."""
import json, subprocess, sys, threading, time, urllib.request

base = "http://127.0.0.1:8000"
target = int(sys.argv[1]) if len(sys.argv) > 1 else 100000

def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return out

def chat():
    body = {"messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": target, "temperature": 0, "thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return time.perf_counter() - t0, d["usage"], d["choices"][0]["message"]["content"][:20]

print(f"VRAM prima: {vram()}")
res = {}
errs = []
def worker(k):
    try:
        res[k] = chat()
    except Exception as e:
        errs.append(f"{k}: {e}")
ts = [threading.Thread(target=worker, args=(k,)) for k in (1, 2)]
t0 = time.perf_counter()
[t.start() for t in ts]; [t.join() for t in ts]
wall = time.perf_counter() - t0
for k in sorted(res):
    dt, usage, txt = res[k]
    print(f"  slot {k}: {dt:.1f}s | prompt {usage['prompt_tokens']} + gen {usage['completion_tokens']} | {txt!r}")
for e in errs:
    print(f"  ERRORE: {e}")
print(f"VRAM dopo : {vram()}   (target slot ~{target}, wall {wall:.1f}s)")
