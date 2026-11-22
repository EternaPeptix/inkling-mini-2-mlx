"""Exact gate: candidate bench JSON vs baseline JSON. Greedy tokens must be identical for every (passes, prompt)."""
import json, sys
b, c = (json.load(open(p)) for p in sys.argv[1:3]); ok = True
for k, v in b["decode"].items():
    w = c["decode"].get(k)
    if w is None: print("MISSING", k); ok = False; continue
    same = v["tokens"] == w["tokens"]; first = next((i for i, (x, y) in enumerate(zip(v["tokens"], w["tokens"])) if x != y), None)
    print("%-10s %s %s speed %.2f -> %.2f tok/s (%+.1f%%)" % (k, "EXACT" if same else "DIFF at token %s" % first, v["digest"], v["tok_s_median"], w["tok_s_median"], 100 * (w["tok_s_median"] / v["tok_s_median"] - 1)))
    ok &= same
print("GATE", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
