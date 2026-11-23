"""Per-kernel GPU durations of steady-state batched decode steps from a Metal System Trace export taken with
MLX_MAX_OPS_PER_BUFFER=0 (one command buffer per kernel).  Steps are delimited by the lm_head kernel (the longest).
Prints the kernel sequence of one step folded by visit (visit start = a kernel pattern repeat) and a duration histogram."""
import re, statistics, sys
def us(x):
    v, u = x.replace(",", "").split()
    return float(v) * {"ns": 1e-3, "µs": 1, "ms": 1e3, "s": 1e6}[u]
def ts(x):
    m, rest = x.split(":"); s, ms, u = rest.split(".")
    return (int(m) * 60 + int(s)) * 1e6 + int(ms) * 1e3 + int(u)
s = open(sys.argv[1]).read()
ids = {m.group(2): m.group(3) for m in re.finditer(r'<([a-z0-9-]+) id="(\d+)" fmt="([^"]*)"', s)}
T, D = [], []
for r in s.split("<row>")[1:]:
    d = {}
    for tag, i, fmt, ref in re.findall(r'<([a-z0-9-]+) (?:id="(\d+)" fmt="([^"]*)"|ref="(\d+)")', r):
        d.setdefault(tag, fmt if i else ids.get(ref, ""))
    if "python" in d.get("process", "") and d.get("gpu-channel-name") == "Compute":
        T.append(ts(d["start-time"])); D.append(us(d["duration"]))
big = sorted(D)[-int(sys.argv[2]) * 3:]
thr = min(big) * 0.9
heads = [i for i, x in enumerate(D) if x >= thr]
heads = [h for h in heads if h > len(D) // 4]
a, b = heads[-3], heads[-2]
S = D[a + 1:b + 1]
span = (T[b] + D[b] - T[a + 1]) / 1e3
print("step kernels %d  GPU sum %.2f ms  span %.2f ms  lm_head %.0f us" % (len(S), sum(S) / 1e3, span, D[b]))
for i in range(0, len(S), 16):
    print("%4d " % i + " ".join("%6.1f" % x for x in S[i:i + 16]))
q = sorted(S)
print("median %.1f  p90 %.1f  sum<15us %.2f ms (%d)  sum 15-40 %.2f ms (%d)  sum>40 %.2f ms (%d)" % (
    statistics.median(S), q[int(.9 * len(q))], sum(x for x in S if x < 15) / 1e3, sum(x < 15 for x in S),
    sum(x for x in S if 15 <= x < 40) / 1e3, sum(15 <= x < 40 for x in S), sum(x for x in S if x >= 40) / 1e3,
    sum(x >= 40 for x in S)))
