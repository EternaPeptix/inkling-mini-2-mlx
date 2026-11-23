import os
"""Attribute GPU intervals of a stage_trace.py Metal System Trace export to stages (see stage_trace.py).

LOG entries are (stage, t_eval_start, t_eval_end) in CPU ns; the trace clock has an unknown offset, fitted by maximizing
the number of GPU intervals that fall inside some eval window."""
import bisect, collections, json, re, sys
def ts(x):
    m, rest = x.split(":"); s, ms, us = rest.split(".")
    return (int(m) * 60 + int(s)) * 1e6 + int(ms) * 1e3 + int(us)
def us(x):
    v, u = x.replace(",", "").split()
    return float(v) * {"ns": 1e-3, "µs": 1, "ms": 1e3, "s": 1e6}[u]
s = open(sys.argv[1]).read(); log = json.load(open(sys.argv[2])); steps = int(sys.argv[3])
ids = {m.group(2): m.group(3) for m in re.finditer(r'<([a-z0-9-]+) id="(\d+)" fmt="([^"]*)"', s)}
T, D = [], []
for r in s.split("<row>")[1:]:
    d = {}
    for tag, i, fmt, ref in re.findall(r'<([a-z0-9-]+) (?:id="(\d+)" fmt="([^"]*)"|ref="(\d+)")', r):
        d.setdefault(tag, fmt if i else ids.get(ref, ""))
    if "python" in d.get("process", "") and d.get("gpu-channel-name") == "Compute":
        T.append(ts(d["start-time"])); D.append(us(d["duration"]))
start = [i for i in range(1, len(T)) if T[i] - T[i - 1] > 300e3][-1]  # the 0.5 s sleep before the timed steps
T, D = T[start:], D[start:]
# groups = GPU intervals separated by idle > 380 us; each LOG window owns >= 1 consecutive groups.  Align the GPU idle gaps
# between groups with the CPU gaps between eval windows (DP: a group boundary either matches the next window
# boundary, cost |gpu gap - cpu gap|, or is a split inside one window, cost = penalty).
groups, cur = [], [0]
for i in range(1, len(T)):
    if T[i] - (T[i - 1] + D[i - 1]) > float(os.environ.get("GAP", "380")):
        groups.append(cur); cur = []
    cur.append(i)
groups.append(cur)
gg = [T[g[0]] - (T[g[0] - 1] + D[g[0] - 1]) for g in groups[1:]]  # gap before group j+1
cg = [(log[i + 1][1] - log[i][2]) / 1e3 for i in range(len(log) - 1)]  # cpu gap before window i+1
G, N, PEN = len(gg), len(cg), 400.0
SIMPLE = len(groups) == len(log)  # long sleep between stage evals: groups map 1:1 to log windows
INF = float("inf")
dp = [[INF] * (N + 1) for _ in range(G + 1)]; bk = [[0] * (N + 1) for _ in range(G + 1)]
dp[0][0] = 0.0
for j in range(G):
    for i in range(max(0, j - 60), min(N, j) + 1):
        v = dp[j][i]
        if v == INF:
            continue
        if i < N and v + abs(gg[j] - cg[i]) < dp[j + 1][i + 1]:
            dp[j + 1][i + 1] = v + abs(gg[j] - cg[i]); bk[j + 1][i + 1] = 1
        if v + PEN < dp[j + 1][i]:
            dp[j + 1][i] = v + PEN; bk[j + 1][i] = 0
if SIMPLE:
    dp = [[0.0] * (N + 1) for _ in range(G + 1)]
elif dp[G][N] == INF:
    sys.exit("alignment failed: groups %d windows %d" % (G + 1, N + 1))
owner = [0] * (G + 1); j, i = G, N
if SIMPLE:
    owner = list(range(G + 1)); j = 0
while j > 0:
    owner[j] = i
    i -= bk[j][i]; j -= 1
print("groups %d windows %d splits %d mean |gap err| %.0f us" % (G + 1, N + 1, G - N, dp[G][N] / max(N, 1)))
tot = collections.defaultdict(float); calls = collections.Counter(n for n, _, _ in log); nk = collections.defaultdict(int)
for j, g in enumerate(groups):
    n = log[owner[j]][0]
    tot[n] += sum(D[x] for x in g); nk[n] += len(g)
allt = sum(tot.values())
print("%-6s %8s %6s %8s %9s" % ("stage", "ms/step", "share", "kern/st", "us/call"))
for n in sorted(tot, key=lambda n: -tot[n]):
    print("%-6s %8.2f %5.1f%% %8.1f %9.1f" % (n, tot[n] / steps / 1e3, 100 * tot[n] / allt, nk[n] / steps,
                                             tot[n] / max(calls[n], 1)))
print("total GPU busy %.2f ms/step" % (allt / steps / 1e3))
if len(sys.argv) > 4:  # print per-stage kernel durations for log entries [a, b)
    a, b = map(int, sys.argv[4].split(":"))
    for j, g in enumerate(groups):
        if a <= owner[j] < b:
            print("%4d %-6s %s" % (owner[j], log[owner[j]][0], " ".join("%.1f" % D[x] for x in g)))
