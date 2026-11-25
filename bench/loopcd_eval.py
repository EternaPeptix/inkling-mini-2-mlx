"""Evaluate opt-in LoopCD quality and speed on loopling-p1/p2/p3/p5.

Usage:
    python bench/loopcd_eval.py --passes 2,3,5,1 --tasks speed,gen,mcq --out bench/loopcd_eval_results.json
"""
import argparse
import collections
import copy
import json
import math
import os
import random
import re
import statistics
import time

import mlx_lm  # first: applies LOOPLING_CMDBUF before mlx.core initializes
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.inkling2 import batchdecode as BD
from mlx_lm.models.inkling2 import loopcd

LETTERS = "ABCDEFGHIJKLMNOP"
PREFIX = "<|message_model|><|content_text|>"
SUFFIX = "\n</state>\n\nAnswer with the letter of the correct option."
SENT = "\x00STATE\x00"
ROUTE_Q = (
    "Classify this task, which is being handed to an AI agent, by what the agent must deliver. "
    "If the task mixes these, classify it by its final deliverable. Mentions of code, repos, or a programming "
    "language do not make a task IMPLEMENT; a review or audit of code is REASON."
)
ROUTE_OPTS = [
    "IMPLEMENT: the deliverable is new or changed code, tests, configs, scripts, or assets (writing, fixing, refactoring, porting, merging, building, installing; includes 'find the cause and fix it' and applying given review comments).",
    "REASON: the deliverable is a judgment or a plan (review, verify, audit, critique, grade, decide, compare, design, plan, prove, or diagnose a root cause without fixing it).",
    "LOOKUP: the deliverable is information gathered with little judgment (search code or list locations, fetch or read a page or docs, summarize a file, collect facts, run something and report its output).",
]
ROUTE_LABELS = ["IMPL", "REASON", "LOOKUP"]

GREEDY = lambda lp: mx.argmax(lp, axis=-1)


def chat_wrap(tok, user: str) -> str:
    try:
        t = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        t = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return t + PREFIX


def question_block(instructions: str, options: list[str]) -> str:
    return (
        "Question: "
        + instructions.strip()
        + "\nOptions:\n"
        + "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(options))
        + "\n\n<state>\n"
    )


def get_letter_ids(tok) -> list[list[int]]:
    out = []
    for ch in LETTERS:
        out.append(
            sorted(
                {
                    tok.encode(ch, add_special_tokens=False)[0],
                    tok.encode(" " + ch, add_special_tokens=False)[0],
                }
            )
        )
    return out


def probs_from_logits(logits_1d: mx.array, letter_ids: list[list[int]], n: int) -> list[float]:
    lp = logits_1d.astype(mx.float32)
    g = mx.array([mx.logsumexp(lp[mx.array(letter_ids[j])]).item() for j in range(n)])
    return mx.softmax(g).tolist()


def combine_cached_option_probs(
    rec: dict, letter_ids: list[list[int]], cfg: loopcd.LoopCDConfig
) -> list[float]:
    n = rec["n_opts"]
    z_r = rec["z_r"]
    z_1 = rec["z_1"]
    if not cfg.active or cfg.is_zero_strength:
        w = None
    elif cfg.adaptive:
        w = mx.array(float(cfg.omega_max), mx.float32) * (
            mx.array(1.0, mx.float32) - mx.array(float(rec["margin"]), mx.float32)
        )
    else:
        w = mx.array(float(cfg.omega), mx.float32)
    gs = []
    for j in range(n):
        tids = letter_ids[j]
        r_arr = mx.array([z_r[tid] for tid in tids], mx.float32)
        if w is None:
            v = r_arr
        else:
            ref_arr = mx.array([z_1[tid] for tid in tids], mx.float32)
            v = r_arr + w * (r_arr - ref_arr)
        gs.append(mx.logsumexp(v).item())
    return mx.softmax(mx.array(gs, mx.float32)).tolist()


def load_stratified_od(path: str, limit: int, seed: int = 42) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    valid = []
    for r in rows:
        q = r["question"]
        labels = [str(x) for x in r["labels"]]
        if not (2 <= len(labels) <= 16):
            continue
        if str(r["expected"]) not in labels:
            continue
        valid.append(r)
    if not limit or limit >= len(valid):
        return valid
    by_type = collections.defaultdict(list)
    for r in valid:
        by_type[r["question"].get("type", "other")].append(r)
    rng = random.Random(seed)
    types = sorted(by_type)
    per = limit // len(types)
    rem = limit % len(types)
    selected = []
    for i, t in enumerate(types):
        pool = list(by_type[t])
        rng.shuffle(pool)
        k = min(len(pool), per + (1 if i < rem else 0))
        selected.extend(pool[:k])
    return selected


def prepare_od_items(rows: list[dict], tok) -> list[dict]:
    items = []
    max_chars = 6000
    for r in rows:
        q = r["question"]
        labels = [str(x) for x in r["labels"]]
        t = q.get("type")
        crit = q.get("criteria") or {}
        if isinstance(crit, list):
            opts = (
                [f"{labels[i]}: {crit[i]}" for i in range(len(labels))]
                if len(crit) == len(labels)
                else [str(l) for l in labels]
            )
        elif t == "noul" and labels == ["no", "yes"]:
            opts = ["no: no, the statement does not hold", "yes: yes, the statement holds"]
        else:
            opts = [f"{l}: {crit[l]}" if l in crit else str(l) for l in labels]
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False, indent=1)
        if len(state) > max_chars:
            state = state[: int(max_chars * 0.75)] + "\n[...]\n" + state[-int(max_chars * 0.25) :]
        ids = tok.encode(chat_wrap(tok, question_block(q.get("instructions", ""), opts) + state + SUFFIX), add_special_tokens=False)
        items.append(
            dict(
                ids=ids,
                n_opts=len(opts),
                gold=labels.index(str(r["expected"])),
                qtype=t,
                family=r["family"],
            )
        )
    return items


def prepare_routing_data(tok, cc_path: str, hard_path: str, head: int = 192, tail: int = 64):
    pre, post = chat_wrap(tok, question_block(ROUTE_Q, ROUTE_OPTS) + SENT + SUFFIX).split(SENT)
    pre_ids = tok.encode(pre, add_special_tokens=False)
    post_ids = tok.encode(post, add_special_tokens=False)
    mid = tok.encode("\n[...]\n", add_special_tokens=False)

    def enc_state(state: str) -> list[int]:
        s = tok.encode(state, add_special_tokens=False)
        if len(s) > head + tail:
            s = s[:head] + mid + s[-tail:]
        return s + post_ids

    cc = [(r["label"], enc_state(r["prompt"])) for r in json.load(open(cc_path)) if r["label"] != "AMBIG"]
    hard = [(g, enc_state(b)) for g, b in json.load(open(hard_path))]
    return pre_ids, {"transcripts": cc, "hard": hard}


def score_routing_probs(datasets: dict, probs_by_ds: dict) -> dict:
    out = {}
    for name, items in datasets.items():
        preds = [ROUTE_LABELS[max(range(3), key=lambda j: p[j])] for p in probs_by_ds[name]]
        golds = [g for g, _ in items]
        r_idx = [i for i, g in enumerate(golds) if g == "REASON"]
        n_idx = [i for i, g in enumerate(golds) if g != "REASON"]
        out[name] = dict(
            n=len(items),
            acc3=round(sum(p == g for p, g in zip(preds, golds)) / len(items), 4),
            route_acc=round(sum((p == "REASON") == (g == "REASON") for p, g in zip(preds, golds)) / len(items), 4),
            reason_recall=round(sum(preds[i] == "REASON" for i in r_idx) / max(1, len(r_idx)), 4),
            nonreason_to_small=round(sum(preds[i] == "REASON" for i in n_idx) / max(1, len(n_idx)), 4),
        )
    return out


def score_od_probs(od_items: list[dict], probs_list: list[list[float]]) -> dict:
    by_type = collections.defaultdict(list)
    nll = []
    for item, p in zip(od_items, probs_list):
        gi = item["gold"]
        ok = int(max(range(len(p)), key=lambda j: p[j]) == gi)
        by_type[item["qtype"]].append(ok)
        nll.append(-math.log(max(p[gi], 1e-9)))
    all_ok = [x for v in by_type.values() for x in v]
    return dict(
        n=len(all_ok),
        acc=round(sum(all_ok) / len(all_ok), 4),
        nll=round(sum(nll) / len(nll), 4),
        by_type={k: round(sum(v) / len(v), 4) for k, v in sorted(by_type.items())},
    )


def eval_mcq_for_model(model, tok, P: int, od_items: list[dict], pre_ids: list[int], routing_ds: dict):
    letter_ids = get_letter_ids(tok)
    all_letter_tids = sorted({tid for group in letter_ids for tid in group})
    tids_arr = mx.array(all_letter_tids)
    results = {}

    def run_single_cfg(tag: str, cfg_kw: dict):
        model.set_loopcd(**cfg_kw)
        # Routing (prefix cached)
        base = make_prompt_cache(model)
        model(mx.array([pre_ids]), cache=base)
        mx.eval([c.state for c in base])
        r_probs = {}
        for dname, items in routing_ds.items():
            ps = []
            for _, suf_ids in items:
                c = copy.deepcopy(base)
                h = model.model.language_model(mx.array([suf_ids]), cache=c)
                lg = model._project_softcap(h[:, -1:, :])[0, 0]
                ps.append(probs_from_logits(lg, letter_ids, 3))
            r_probs[dname] = ps
        # OpenDecisions
        od_probs = []
        for item in od_items:
            h = model.model.language_model(mx.array([item["ids"]]))
            logits = model._project_softcap(h[:, -1:, :])[0, 0]
            od_probs.append(probs_from_logits(logits, letter_ids, item["n_opts"]))
        results[tag] = {
            "cfg": cfg_kw,
            "routing": score_routing_probs(routing_ds, r_probs),
            "opendecisions": score_od_probs(od_items, od_probs),
        }
        print(f"[MCQ p{P}] {tag:22s} | routing cc={results[tag]['routing']['transcripts']} "
              f"hard={results[tag]['routing']['hard']} | od={results[tag]['opendecisions']}", flush=True)

    run_single_cfg("off", dict(mode="off"))
    if P == 1:
        return results

    hidden_cfgs = [
        ("hidden_w0.2", dict(mode="hidden", omega=0.2)),
        ("hidden_w0.3", dict(mode="hidden", omega=0.3)),
        ("hidden_w0.5", dict(mode="hidden", omega=0.5)),
        ("hidden_w0.75", dict(mode="hidden", omega=0.75)),
        ("hidden_w1.0", dict(mode="hidden", omega=1.0)),
        ("hidden_bs_w0.5", dict(mode="hidden", omega=0.5, blend_bs=True)),
    ]
    for tag, kw in hidden_cfgs:
        run_single_cfg(tag, kw)

    # LoopCD-Logits sweep: collect exact (z_R, z_1, margin) ONCE via return_ref=True + _logits_pair_softcap!
    model.set_loopcd("logits", omega=0.0)
    base_l = make_prompt_cache(model)
    model(mx.array([pre_ids]), cache=base_l)
    mx.eval([c.state for c in base_l])

    def extract_pair_record(h_r, h_1, n_opts):
        zr, z1 = model._logits_pair_softcap(h_r[:, -1:, :], h_1[:, -1:, :])
        zr1, z11 = zr[0, 0], z1[0, 0]
        lp = zr1 - mx.logsumexp(zr1)
        top2 = mx.exp(mx.topk(lp, 2))
        margin = float(mx.abs(top2[0] - top2[1]).item())
        zr_sub = zr1[tids_arr].tolist()
        z1_sub = z11[tids_arr].tolist()
        return {
            "n_opts": n_opts,
            "margin": margin,
            "z_r": dict(zip(all_letter_tids, zr_sub)),
            "z_1": dict(zip(all_letter_tids, z1_sub)),
        }

    r_recs = {dname: [] for dname in routing_ds}
    for dname, items in routing_ds.items():
        for _, suf_ids in items:
            c = copy.deepcopy(base_l)
            hr, h1 = model.model.language_model(mx.array([suf_ids]), cache=c, return_ref=True)
            r_recs[dname].append(extract_pair_record(hr, h1, 3))

    od_recs = []
    for item in od_items:
        hr, h1 = model.model.language_model(mx.array([item["ids"]]), cache=None, return_ref=True)
        od_recs.append(extract_pair_record(hr, h1, item["n_opts"]))

    logits_cfgs = [
        ("logits_w0.0_chk", dict(mode="logits", omega=0.0, adaptive=False)),
        ("logits_w0.2", dict(mode="logits", omega=0.2, adaptive=False)),
        ("logits_w0.3", dict(mode="logits", omega=0.3, adaptive=False)),
        ("logits_w0.5", dict(mode="logits", omega=0.5, adaptive=False)),
        ("logits_w0.75", dict(mode="logits", omega=0.75, adaptive=False)),
        ("logits_ad_w0.3", dict(mode="logits", adaptive=True, omega_max=0.3)),
        ("logits_ad_w0.5", dict(mode="logits", adaptive=True, omega_max=0.5)),
        ("logits_ad_w0.75", dict(mode="logits", adaptive=True, omega_max=0.75)),
        ("logits_ad_w1.0", dict(mode="logits", adaptive=True, omega_max=1.0)),
        ("logits_ad_w1.5", dict(mode="logits", adaptive=True, omega_max=1.5)),
    ]
    for tag, kw in logits_cfgs:
        cfg = loopcd.LoopCDConfig(**kw)
        r_probs = {
            dname: [combine_cached_option_probs(rec, letter_ids, cfg) for rec in recs]
            for dname, recs in r_recs.items()
        }
        od_probs = [combine_cached_option_probs(rec, letter_ids, cfg) for rec in od_recs]
        res_entry = {
            "cfg": kw,
            "routing": score_routing_probs(routing_ds, r_probs),
            "opendecisions": score_od_probs(od_items, od_probs),
        }
        if tag == "logits_w0.0_chk":
            assert res_entry["routing"] == results["off"]["routing"], "logits w=0 routing mismatch!"
            assert res_entry["opendecisions"]["acc"] == results["off"]["opendecisions"]["acc"], "logits w=0 OD acc mismatch!"
            continue
        results[tag] = res_entry
        print(f"[MCQ p{P}] {tag:22s} | routing cc={res_entry['routing']['transcripts']} "
              f"hard={res_entry['routing']['hard']} | od={res_entry['opendecisions']}", flush=True)

    # End-to-end verification of logits_ad_w0.5 via loopcd.combine_logits on the same T=1 qmv projection
    model.set_loopcd("logits", adaptive=True, omega_max=0.5)
    base_v = make_prompt_cache(model)
    model(mx.array([pre_ids]), cache=base_v)
    mx.eval([c.state for c in base_v])
    v_probs = {}
    for dname, items in routing_ds.items():
        ps = []
        for _, suf_ids in items:
            c = copy.deepcopy(base_v)
            hr, h1 = model.model.language_model(mx.array([suf_ids]), cache=c, return_ref=True)
            zr, z1 = model._logits_pair_softcap(hr[:, -1:, :], h1[:, -1:, :])
            lg = loopcd.combine_logits(zr, z1, model.loopcd)[0, 0]
            ps.append(probs_from_logits(lg, letter_ids, 3))
        v_probs[dname] = ps
    v_res = score_routing_probs(routing_ds, v_probs)
    assert v_res == results["logits_ad_w0.5"]["routing"], f"E2E verify mismatch: {v_res} vs {results['logits_ad_w0.5']['routing']}"
    print(f"[MCQ p{P}] verified logits_ad_w0.5 end-to-end routing exact match", flush=True)
    model.set_loopcd("off")
    return results


# 40 deterministic reasoning / combinatorics / code-tracing problems (answers verified in Python)
GEN_PROBLEMS = [
    # 16 multi-step arithmetic / word problems
    ("A bookstore sells notebooks for $7 each and pens for $3 each. Maya buys 8 notebooks and some pens for a total of $95. How many pens did she buy?", "13"),
    ("A rectangular garden is 18 meters long and 12 meters wide. A uniform concrete path of width 2 meters is built completely inside the garden along its four edges, reducing the planted inner rectangle. What is the area in square meters of the concrete path?", "104"),
    ("A tank has 360 liters of water. Pipe A fills it at 18 liters per minute while drain B empties it at 12 liters per minute. Starting from 360 liters in a 900-liter capacity tank with both open, how many minutes until the tank is completely full?", "90"),
    ("Liam scored 78, 84, 91, and 87 on his first four tests. What minimum score must he get on his fifth test so that his five-test average is 88?", "100"),
    ("A shopkeeper marks up an item by 40% above its $150 cost, and later offers a 25% discount off the marked price. What is the final selling price in dollars?", "157.5"),
    ("Two trains start 455 km apart and travel toward each other. One travels at 60 km/h and the other at 70 km/h. After 2.5 hours, how many kilometers apart are they?", "130"),
    ("Sam is 4 times as old as Ava today. In 6 years, Sam will be 2.5 times as old as Ava. How old is Sam today?", "24"),
    ("Worker A can finish a job alone in 12 hours, and Worker B alone in 18 hours. They work together for 4 hours, then Worker A leaves. How many more hours does Worker B need to finish the remaining job?", "8"),
    ("A fruit vendor buys 120 apples for $48 total. During transport, 20 apples spoil and cannot be sold. At what price in cents per good apple must he sell the rest to make a 25% profit on his $48 cost?", "60"),
    ("A car drives 120 km at 40 km/h and then 180 km at 60 km/h. What is its average speed in km/h for the entire 300 km trip?", "50"),
    ("If 3/5 of a number plus 18 equals 63, what is twice that number?", "150"),
    ("A jar contains red, blue, and green marbles in the ratio 3:5:7. If there are 48 more green marbles than red marbles, how many total marbles are in the jar?", "180"),
    ("An alloy of 40 kg is 30% copper. How many kilograms of pure copper must be added so the resulting alloy is 50% copper?", "16"),
    ("A staircase has 10 steps. You can climb either 1 step or 2 steps at a time. How many distinct ways are there to reach step 10 from step 0?", "89"),
    ("Ben spent 1/3 of his money on rent, then 1/4 of the remaining money on food, and had $180 left. How many dollars did he start with?", "360"),
    ("Three consecutive odd integers have a sum of 141. What is the largest of the three integers?", "49"),
    # 12 logic / number theory / combinatorics
    ("What is the remainder when 3^20 is divided by 7?", "2"),
    ("How many positive integers from 1 to 200 inclusive are divisible by 3 or 5, but not by both?", "80"),
    ("In a room of 11 people, everyone shakes hands with everyone else except their own spouse; there are 3 married couples and 5 single people. How many total handshakes occur?", "52"),
    ("What is the sum of all positive divisors of 72?", "195"),
    ("How many distinct 4-digit integers can be formed using the digits 1, 2, 3, 4, 5 without repetition that are divisible by 5?", "24"),
    ("The 3rd term of an arithmetic sequence is 14 and the 8th term is 39. What is the sum of the first 10 terms?", "265"),
    ("Two fair six-sided dice are rolled. What is the probability that the sum is a prime number? Express your answer as a reduced fraction a/b.", "5/12"),
    ("On an island, Knights always tell the truth and Knaves always lie. Alice says: 'Both Bob and I are Knaves.' Bob says nothing. What are Alice and Bob? Answer as two words separated by a comma, e.g. Knight, Knave.", "Knave, Knight"),
    ("How many trailing zeros are in 65! (65 factorial)?", "15"),
    ("Find the smallest positive integer n such that n % 3 == 2, n % 5 == 3, and n % 7 == 2.", "23"),
    ("A palindrome reads the same forwards and backwards. How many 4-digit positive integers are palindromes?", "90"),
    ("What is the greatest common divisor of 462 and 1071?", "21"),
    # 12 code-tracing / symbolic reasoning
    ("What value does this Python snippet print?\nx = 0\nfor i in range(1, 7):\n    if i % 2 == 0:\n        x += i * i\n    else:\n        x -= i\nprint(x)", "47"),
    ("What value does this Python function return for f(5)?\ndef f(n):\n    if n <= 1:\n        return 1\n    return n + f(n - 1) + f(n - 2)", "29"),
    ("What does this Python code print?\na = [2, 5, 1, 8, 3]\ns = 0\nfor i in range(len(a)):\n    for j in range(i + 1, len(a)):\n        if a[i] < a[j]:\n            s += 1\nprint(s)", "6"),
    ("What is the output of this Python snippet?\nn = 1234\nr = 0\nwhile n > 0:\n    r = r * 10 + (n % 10)\n    n //= 10\nprint(r - 432)", "3889"),
    ("What does this Python snippet print?\nvals = [1, 1]\nfor _ in range(6):\n    vals.append((vals[-1] + 2 * vals[-2]) % 10)\nprint(vals[-1])", "5"),
    ("What does this Python snippet print?\ns = 0\nfor k in range(1, 16):\n    if k % 3 == 0:\n        continue\n    if k > 11:\n        break\n    s += k\nprint(s)", "48"),
    ("What does this Python snippet print?\nd = {}\nfor w in ['ab', 'bc', 'ab', 'cd', 'bc', 'ab']:\n    d[w] = d.get(w, 0) + len(w)\nprint(d['ab'] * 10 + d['bc'])", "64"),
    ("What does this Python snippet print?\na, b = 12, 5\nprint((a ^ b) + (a & b) * 2)", "17"),
    ("What does this Python snippet print?\ndef g(xs):\n    return [x * x for x in xs if x % 3 != 0]\nprint(sum(g([2, 3, 4, 6, 5])))", "45"),
    ("What does this Python snippet print?\ntotal = 0\nfor i in range(4):\n    for j in range(4):\n        if (i + j) % 2 == 0 and i <= j:\n            total += i + j\nprint(total)", "18"),
    ("What does this Python snippet print?\nx = 100\nfor step in [15, -20, 35, -10, 25]:\n    x = max(90, x + step)\nprint(x)", "145"),
    ("What does this Python snippet print?\nnums = [3, 1, 4, 1, 5, 9]\nres = 0\nfor v in sorted(set(nums)):\n    res = res * 2 + v\nprint(res)", "75"),
]


def _norm_ans(s: str) -> str:
    s = s.strip().rstrip(".")
    s = s.replace("$", "").replace(",", ", ") if "knave" in s.lower() or "knight" in s.lower() else s.replace("$", "").replace(",", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


def extract_answer(text: str, gold: str = "") -> str:
    if "<|message_model|><|content_text|>" in text:
        after = text.rsplit("<|message_model|><|content_text|>", 1)[-1]
        if re.sub(r"<\|[^|]+\|>", "", after).strip():
            text = after
    text = re.sub(r"<\|[^|]+\|>", "", text).strip()
    matches = [
        m.group(1).strip()
        for m in re.finditer(r"Final Answer\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if "<answer>" not in m.group(1).lower() and "<value>" not in m.group(1).lower()
    ]
    ans = ""
    if matches:
        ans = matches[-1]
    else:
        bm = list(re.finditer(r"\\boxed\{([^}]+)\}", text))
        if bm:
            ans = bm[-1].group(1).strip()
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                ans = lines[-1]
    ans = re.sub(r"^\*\*|\*\*$", "", ans).strip()
    ans = re.sub(r"^\\boxed\{(.+)\}$", r"\1", ans).strip()
    if gold and re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?", gold):
        nm = re.search(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?", ans.replace("$", ""))
        if nm:
            ans = nm.group(0)
    return _norm_ans(ans)


def rep4_rate(tok_ids: list[int]) -> float:
    if len(tok_ids) < 8:
        return 0.0
    grams = [tuple(tok_ids[i : i + 4]) for i in range(len(tok_ids) - 3)]
    return 1.0 - (len(set(grams)) / len(grams))


def prefill_batch(model, ids: list[int]):
    c = make_prompt_cache(model)
    p = mx.array(ids)
    while p.size > 1:
        n = min(2048, p.size - 1)
        model(p[:n][None], cache=c)
        mx.eval([x.state for x in c])
        p = p[n:]
    return c


def eval_gen_for_model(model, tok, P: int, max_tokens: int):
    eos_ids = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    eos_ids.add(tok.eos_token_id)
    prompts = [
        chat_wrap(
            tok,
            q + "\n\nSolve concisely step by step, then end with a single line: Final Answer: <value>",
        )
        for q, _ in GEN_PROBLEMS
    ]
    golds = [_norm_ans(a) for _, a in GEN_PROBLEMS]
    enc = [tok.encode(p, add_special_tokens=False) for p in prompts]

    cfgs = [("off", dict(mode="off"))]
    if P > 1:
        cfgs += [
            ("hidden_w0.2", dict(mode="hidden", omega=0.2)),
            ("hidden_w0.3", dict(mode="hidden", omega=0.3)),
            ("hidden_w0.5", dict(mode="hidden", omega=0.5)),
            ("hidden_w0.75", dict(mode="hidden", omega=0.75)),
            ("hidden_bs_w0.5", dict(mode="hidden", omega=0.5, blend_bs=True)),
            ("logits_w0.2", dict(mode="logits", omega=0.2, adaptive=False)),
            ("logits_w0.3", dict(mode="logits", omega=0.3, adaptive=False)),
            ("logits_w0.5", dict(mode="logits", omega=0.5, adaptive=False)),
            ("logits_ad_w0.3", dict(mode="logits", adaptive=True, omega_max=0.3)),
            ("logits_ad_w0.5", dict(mode="logits", adaptive=True, omega_max=0.5)),
            ("logits_ad_w0.75", dict(mode="logits", adaptive=True, omega_max=0.75)),
            ("logits_ad_w1.0", dict(mode="logits", adaptive=True, omega_max=1.0)),
        ]

    out = {}
    for tag, kw in cfgs:
        model.set_loopcd(**kw)
        all_gen = [[] for _ in range(len(enc))]
        finished = [False] * len(enc)
        for start in range(0, len(enc), 8):
            batch_ids = enc[start : start + 8]
            C = len(batch_ids)
            caches = [prefill_batch(model, ids) for ids in batch_ids]
            dec = BD.BatchDecoder(model, caches)
            y = mx.array([ids[-1] for ids in batch_ids])
            y = dec.step(y)
            mx.async_eval(y)
            for step in range(max_tokens):
                ny = None
                if step + 1 < max_tokens:
                    ny = dec.step(y)
                    mx.async_eval(ny)
                row_toks = y.tolist()
                all_done = True
                for i, tv in enumerate(row_toks):
                    idx = start + i
                    if not finished[idx]:
                        if tv in eos_ids:
                            finished[idx] = True
                        else:
                            all_gen[idx].append(tv)
                    if not finished[idx]:
                        all_done = False
                y = ny
                if all_done or y is None:
                    mx.synchronize()
                    break
            del dec, caches
            mx.clear_cache()

        correct = []
        reps = []
        lens = []
        samples = {}
        for idx, (g_toks, gold) in enumerate(zip(all_gen, golds)):
            txt = tok.decode(g_toks)
            pred = extract_answer(txt, gold=gold)
            ok = int(pred == gold)
            correct.append(ok)
            reps.append(rep4_rate(g_toks))
            lens.append(len(g_toks))
            if idx in (2, 13, 16, 28, 30):
                samples[idx] = {"ok": bool(ok), "pred": pred, "gold": gold, "tail": txt[-160:].replace("\n", " ")}

        out[tag] = {
            "cfg": kw,
            "n": len(correct),
            "acc": round(sum(correct) / len(correct), 4),
            "correct": sum(correct),
            "arith_acc": round(sum(correct[:16]) / 16, 4),
            "logic_acc": round(sum(correct[16:28]) / 12, 4),
            "code_acc": round(sum(correct[28:40]) / 12, 4),
            "eos_rate": round(sum(finished) / len(finished), 4),
            "mean_rep4": round(sum(reps) / len(reps), 4),
            "mean_len": round(sum(lens) / len(lens), 1),
            "correct_mask": correct,
            "samples": samples,
        }
        print(
            f"[GEN p{P}] {tag:18s} | acc={out[tag]['acc']:.3f} ({out[tag]['correct']}/40) "
            f"[A:{out[tag]['arith_acc']:.2f} L:{out[tag]['logic_acc']:.2f} C:{out[tag]['code_acc']:.2f}] "
            f"eos={out[tag]['eos_rate']:.2f} rep4={out[tag]['mean_rep4']:.3f} len={out[tag]['mean_len']:.1f}",
            flush=True,
        )
    model.set_loopcd("off")
    return out


def eval_speed_for_model(model, tok, P: int, tokens: int = 128, warm: int = 4):
    prompts = [
        "Explain how B-tree insertion works with a node split example.",
        "Prove that the sum of the first n odd numbers is n squared.",
        "Write a Python parser for ISO-8601 durations.",
        "Summarize the causes of the French Revolution.",
        "Implement binary search cleanly and discuss edge cases.",
        "Explain the water cycle and three common misconceptions.",
        "Write a shell script to find the ten largest files.",
        "Compare quicksort and mergesort space and time trade-offs.",
    ]
    ids_list = [tok.encode(chat_wrap(tok, p), add_special_tokens=False) for p in prompts]
    cfgs = [("off", dict(mode="off"))]
    if P > 1:
        cfgs += [
            ("hidden_w0.5", dict(mode="hidden", omega=0.5)),
            ("logits_w0.3", dict(mode="logits", omega=0.3, adaptive=False)),
            ("logits_ad_w0.5", dict(mode="logits", adaptive=True, omega_max=0.5)),
        ]
    res = {}
    prefill_ids = [(17 * i * i + 3 * i) % 40000 + 1000 for i in range(512)]
    p_arr = mx.array([prefill_ids])

    for tag, kw in cfgs:
        model.set_loopcd(**kw)
        # generate_step C=1
        gs_runs = []
        for _ in range(3):
            cnt = 0
            t0 = None
            for _, _ in generate_step(mx.array(ids_list[0]), model, max_tokens=tokens, sampler=GREEDY):
                cnt += 1
                if cnt == warm:
                    t0 = time.perf_counter()
            gs_runs.append((tokens - warm) / (time.perf_counter() - t0))
        gs_tps = statistics.median(gs_runs)

        # BatchDecoder C=1 & C=8
        bd_tps = {}
        bd_ms = {}
        for C in (1, 8):
            runs_tps = []
            for _ in range(3):
                caches = [prefill_batch(model, ids_list[i]) for i in range(C)]
                dec = BD.BatchDecoder(model, caches)
                y = mx.array([ids_list[i][-1] for i in range(C)])
                y = dec.step(y)
                mx.async_eval(y)
                t0 = None
                for n in range(tokens):
                    ny = None
                    if n + 1 < tokens:
                        ny = dec.step(y)
                        mx.async_eval(ny)
                    _ = y.tolist()
                    if n + 1 == warm:
                        t0 = time.perf_counter()
                    y = ny
                dt = time.perf_counter() - t0
                runs_tps.append(C * (tokens - warm) / dt)
                del dec, caches
                mx.clear_cache()
            med = statistics.median(runs_tps)
            bd_tps[C] = round(med, 2)
            bd_ms[C] = round(1000.0 * C / med, 2)

        # Prefill T=512
        # warmup
        c_w = make_prompt_cache(model)
        z_w = model(p_arr, cache=c_w)
        mx.eval(z_w)
        pf_runs = []
        for _ in range(3):
            c_p = make_prompt_cache(model)
            t0 = time.perf_counter()
            z_p = model(p_arr, cache=c_p)
            mx.eval(z_p)
            pf_runs.append(512.0 / (time.perf_counter() - t0))

        res[tag] = {
            "cfg": kw,
            "generate_step_c1_tok_s": round(gs_tps, 2),
            "bd_c1_tok_s": bd_tps[1],
            "bd_c1_ms": bd_ms[1],
            "bd_c8_agg_tok_s": bd_tps[8],
            "bd_c8_ms": bd_ms[8],
            "prefill_512_tok_s": round(statistics.median(pf_runs), 1),
        }
        print(
            f"[SPD p{P}] {tag:16s} | gs_c1={res[tag]['generate_step_c1_tok_s']:6.1f} "
            f"bd_c1={res[tag]['bd_c1_tok_s']:6.1f} ({res[tag]['bd_c1_ms']:5.2f}ms) | "
            f"bd_c8={res[tag]['bd_c8_agg_tok_s']:6.1f} ({res[tag]['bd_c8_ms']:5.2f}ms) | "
            f"pf512={res[tag]['prefill_512_tok_s']:6.1f}",
            flush=True,
        )
    model.set_loopcd("off")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.expanduser("~/loopling/models"))
    ap.add_argument("--passes", default="2,3,5,1")
    ap.add_argument("--tasks", default="speed,gen,mcq")
    ap.add_argument("--od-path", default=os.path.expanduser("~/loopling/data/opendecisions.jsonl"))
    ap.add_argument("--cc-path", default=os.path.expanduser("~/loopling/eval_cc.json"))
    ap.add_argument("--hard-path", default=os.path.expanduser("~/loopling/eval_hard.json"))
    ap.add_argument("--od-limit", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="bench/loopcd_eval_results.json")
    a = ap.parse_args()

    tasks = set(t.strip().lower() for t in a.tasks.split(",") if t.strip())
    passes = [int(x) for x in a.passes.split(",") if x.strip()]

    results = {}
    if os.path.exists(a.out):
        try:
            results = json.load(open(a.out))
        except Exception:
            results = {}

    od_rows = load_stratified_od(a.od_path, a.od_limit) if "mcq" in tasks else []

    for P in passes:
        key = f"p{P}"
        entry = results.setdefault(key, {})
        model, tok = load(os.path.join(a.models_dir, f"loopling-p{P}"))
        mx.eval(model.parameters())

        if "speed" in tasks:
            entry["speed"] = eval_speed_for_model(model, tok, P)
        if "gen" in tasks:
            entry["gen"] = eval_gen_for_model(model, tok, P, max_tokens=a.max_tokens)
        if "mcq" in tasks:
            od_items = prepare_od_items(od_rows, tok)
            pre_ids, routing_ds = prepare_routing_data(tok, a.cc_path, a.hard_path)
            entry["mcq"] = eval_mcq_for_model(model, tok, P, od_items, pre_ids, routing_ds)

        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)
        del model
        mx.clear_cache()


if __name__ == "__main__":
    main()
