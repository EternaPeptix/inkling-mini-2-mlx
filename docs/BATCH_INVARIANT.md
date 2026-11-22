# Batch-invariant T-row decode forward (lane D, `LOOPLING_BATCH_INVARIANT=1`, default off)

Goal: a T-token forward (T = 2..8, batch 1, existing cache, decode time) whose per-row logits and cache writes are
bitwise identical to T sequential 1-token forwards, so speculative verification (selfspec) can pass the exact gate.
Code: `vendor/mlx_lm/models/inkling2/batch_invariant.py` (class-level wrappers installed from `inkling2/__init__.py`
when the flag is set; they fall through to the original code unless `batch_invariant.active()` is entered, which only
`selfspec.py` does around the verify forward, so prefill chunks of 2..8 tokens are untouched). Tools:
`bench/bi_compare.py` (per-op trace of T-row vs T x 1-token, flags the first op whose rows differ with equal inputs),
`bench/bi_micro.py` (which formulation of one op is bitwise equal), `bench/bi_time.py`, `bench/bi_matrix.sh`, `bench/bi_gate.sh`.
`LOOPLING_BI_OPS=qmm,mm,router,conv,moe,attn` enables sources one by one (diagnostics).

## Divergence sources, fixes, costs (T=4, p3, ms per forward; default T-row forward 42.6 ms)

Which ops are the 1-token vs T-row difference was found by tracing: the baseline 1-token path differs from every
multi-row path (block sizes 2, 4, 8 agree with each other; lane C), as expected from M=1 kernel selection.

| source | what differs | fix | cost |
|---|---|---|---|
| attention | L>1 uses the custom half-precision `relative_attention` kernel; L=1 uses the SDPA vector kernel with a bias mask; plus the rotating-cache layout (concat path vs in-place ring) | per row: cache write with a 1-token in-place update and `fastdecode._core` (the exact 1-token attention core, masks cached per cache state); the rest of the attention prologue stays batched (elementwise, verified invariant) | +8.4 ms (largest) |
| float / bf16 `nn.Linear` (message gates, router matmul) | M>1 takes the gemm kernel, M=1 the gemv. `x.reshape(T,1,K) @ W` is folded back into M; broadcasting W did not help either. bf16 `down_proj` also differs although a single micro-test said no (micro-tests can lie; only the full trace is trusted) | one `x[:, j] @ W.T` per row (T tiny gemv kernels) | +3.3 ms |
| router logits (`hidden.float() @ W.T`, 256 experts) | same gemm vs gemv | per-row matmul, then the (row-independent) routing kernel | +1.6 ms |
| affine 8-bit `QuantizedLinear` (o_proj, lm_head; small-N projections happen to agree) | qmm vs qmv. Reshape to (T,1,K) is folded into M; `gather_qmm` differs | `quantized_matmul` with x as a batch of T (T,1,K) against the weights broadcast with stride 0 to a batch of T: each row is a real M=1 qmv. Weights are streamed from cache, not re-fetched for small matrices; lm_head (463 MB) is read T times | +1.7 ms |
| MoE `gather_qmm` (NVFP4 / MXFP8 experts) | the unsorted path with batch rows already matches; the sorted path (`indices.size >= 64`, T=8 with top-8) uses another kernel | never sort in the T-row path | +0.6 ms |
| depthwise short convs (`Conv1d`, 4 per layer) | T output positions in one conv != T conv calls of length 1 (wrapper showed Conv1d as a source in every layer) | T windows of length K stacked as a batch (T,K,C), one conv call; same kernel as decode | ~0 |
| norms, L2 clips, softcapped SwiGLU, message-gate norms, depth attention kernel, routing kernel, dynamic conv, embedding, logit softcap | already per-row invariant (row-wise reductions, one threadgroup per row) | none | 0 |
| final `logsumexp`/`argmax` over the 201k vocabulary (in selfspec) | reduction strategy may depend on the row count | per-row call | ~0 |
| rotating-cache rollback (selfspec) | in-place ring writes of rejected rows overwrite old window entries | snapshot (alias) before the verify, restore, replay the kept rows' cache writes (`batch_invariant` stores the per-row k/v) | ~0 |

Total at T=4, p3: 51.9 ms invariant vs 44.4 default T-row vs 108.8 for 4 x 1-token (reference path) or 67.8 ms with
lane A's compiled decode (`LOOPLING_COMPILE_DECODE=1`).

## Measured: T-row forward time (ms, medians of 12, p3, ctx 300)

| T | 1-token x T (ref) | 1-token x T (lane A fast) | default T-row | invariant T-row |
|---|---|---|---|---|
| 2 | 52.0 | 40.8 | 36.7 | 38.8 |
| 3 | 80.1 | 60.3 | 40.5 | 44.9 |
| 4 | 108.8 | 67.8 | 44.4 | 51.9 |
| 5 | 137.5 | 87.3 | 48.8 | 59.5 |

(single 1-token forward: 28.4 ms reference, 20.3 ms fast path.) Decode is bound by primitive count, not bytes
(docs/PROFILE.md), so per-row ops cost about what a 1-token step costs per op; the weights are still read once for
the large expert/attention matmuls.

## Per-row bitwise result

`bench/bi_compare.py` compares final logits of every row and the last-T KV entries plus conv states of every layer
visit against T sequential 1-token forwards from the same cache state: bitwise equal (0 mismatching cache entries)
for T = 2..8 (p5, sliding window forced to 48 so the ring is saturated), T=4/5 at p3 with 300 and 3000 context,
T=3 with window 64, T=2 at p1; with lane A's flags (`LOOPLING_COMPILE_DECODE=1 LOOPLING_FUSED_CONV=1`) off and on
(one target satisfies both). See `bench/bi_matrix.sh`.

## Selfspec under the flag (128 tokens, 3 prompts, draft 1 pass; exact = tokens equal the baseline run in-process)

Exact on every configuration (see the gate results in the lane report). Speed is in the lane report: with the
reference decode path selfspec is faster on the high-acceptance prompt (math, up to 1.46x at p5) and prose at p5, slower on
code/prose at p3; with lane A's fast path the 1-token baseline gets 40% faster and selfspec is slower on all but p5 math.

## Route 2 (decode through the L>1 kernels): not pursued

Sending the 1-token decode through the L>1 branch would make verify rows trivially equal to decode but changes the
baseline tokens (the reference is the 1-token path), forfeits lane A's gain, and the custom half-precision attention
kernel is a different numerical function. Lane C's measurements (block 2/4/8 bit-identical, L=1 the odd one out) are
consistent with this analysis.

## Not done / ideas

* The per-row ops are launch-bound; a single custom kernel reproducing the qmv summation order for T rows (reading
  weights once) or a compiled per-row segment (as lane A does for 1-token) would cut the 8-9 ms/row overhead.
* `T > 8` is not covered (falls through to the default path).
* Batch > 1 is not covered.
