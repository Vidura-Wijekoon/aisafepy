# The hashed n-gram classifier

`aisafepy.stream.hashed_classifier.HashedNGramClassifierGuard` is a
pure-Python text classifier that the `adapt` compiler can train and the
canary controller can deploy without `torch`, a GPU, or a model
download. This page explains what it is, when to reach for it instead
of a distilled transformer, what it needs to learn anything useful, and
where it stops being useful.

## What it is

A logistic-regression model over hashed n-grams:

| Step | Detail |
|---|---|
| Normalise | `normalize_for_matching`: NFKC, zero-width and TAG characters removed, Cyrillic / Greek / fullwidth confusables folded onto Latin, casefold. Homoglyph and spacing tricks collapse onto the same tokens before the model sees them. |
| Featurise | Word unigrams, word bigrams, character 3- to 5-grams. Each feature string is hashed with CRC32 into one of `n_buckets` slots (2^18 by default); a second hash picks a sign so collisions cancel in expectation. Vectors are L2-normalised. |
| Train | Mini-batch SGD with per-feature AdaGrad step sizes, L2 regularisation applied to touched weights, inverse-frequency class weights, early stopping on a held-out split, deterministic given `seed`. |
| Explain | During training one example n-gram per bucket is remembered (bounded), so `explain(text)` returns the n-grams that pushed a decision, with weights. |
| Persist | `save()` writes JSON with only the non-zero weights plus the training report; `load()` restores an identical model. |

Inference is a sparse dot product: tens of microseconds for a short
prompt, well inside the Tier-2 band.

## When to use it

Use it when a red-team cluster shares surface vocabulary and you need
something deployable *today*:

- "ignore previous instructions" variants, role-hijack phrasings,
  a specific tool name or product string that keeps appearing in
  attacks against your agent;
- obfuscated repeats of the above (the normaliser does the heavy
  lifting);
- any cluster where the regex target produced a long list of brittle
  literals and you would rather have one weighted model.

Prefer the transformer distillation target when the cluster is about
*meaning* rather than wording: paraphrased requests, multi-step
social engineering, content categories (self-harm, hate), or anything
where a human would need to read the sentence rather than scan it.

The two are not exclusive. A common shape is the hashed classifier
live in Tier 2 within an hour of the red-team run, and the distilled
transformer replacing it a week later once the GPU job has finished
and the canary shows a better FPR.

## Training data

| Need | Why |
|---|---|
| At least 8 positives (`min_positives`) | Fewer and the model memorises the examples; the target defers instead. |
| At least 8 negatives (`min_negatives`) | Without a benign pool the model learns "all text is unsafe". The target uses, in order: negatives passed by the compiler, negatives given at construction, the cluster's own non-violation records. |
| Negatives that look like real traffic | The false-positive rate on the held-out split is only meaningful if the negatives resemble what the guard will see. Pull them from production allows (`DecisionLogSource(..., only_violations=False)`) rather than from a different domain. |
| Dozens, not thousands | Linear models over hashed n-grams saturate quickly. More data helps precision; it does not add understanding. |

The benchmark `benchmarks/bench_hashed_classifier.py` trains on a 70%
stratified split of `benchmarks/datasets/injection_eval.jsonl` and
reports the holdout metrics; run it to calibrate expectations before
trusting the compiler's validation numbers on your own clusters.

## In the compile loop

```python
from aisafepy.adapt import GuardCompiler, JSONLSource, PyRITSource, Target

compiler = GuardCompiler(
    source=PyRITSource(jsonl_path="pyrit_runs.jsonl"),
    negatives=JSONLSource("benign_traffic.jsonl", only_violations=False),
    targets=[Target.hashed_classifier(out_dir="guards/hashed", threshold=0.6)],
)
report = compiler.compile()
```

Each cluster that clears the minimums yields a `hashed_classifier`
artifact whose payload is:

```json
{"path": "guards/hashed/hashed_cluster_3.json",
 "metrics": {"precision": 0.96, "recall": 0.91, "f1": 0.93,
             "false_positive_rate": 0.02, "n_validation": 14, ...},
 "threshold": 0.6}
```

`CanaryController.start()` materialises the artifact with
`materialize_hashed_classifier` and places the guard in Tier 2 (shadow
or live). The payload metrics are the pre-traffic estimate; the canary's
own false-positive accounting against labelled benign traffic is what
decides promotion or rollback.

Closing the loop from production: `OTLPSpanSource` and
`DecisionLogSource` (in `aisafepy.adapt.sources_otlp`) read AIsafePy's
own spans and JSONL decision logs back into `FailureRecord`s, so the
texts a Tier-3 judge or a human reviewer flagged become the next
cluster's positives, and the allows become its negatives.

## Limits

- **No semantics.** It is a bag of n-grams. Negation, paraphrase and
  intent are invisible to it; "do not ignore the instructions" shares
  almost every feature with "ignore the instructions".
- **Vocabulary bound.** It generalises to texts that reuse training
  vocabulary and to obfuscations the normaliser undoes. A new phrasing
  is a new cluster.
- **Collisions.** Hashing means unrelated n-grams can share a bucket.
  Rare in practice at 2^18 buckets, but it makes `explain()`
  indicative rather than causal.
- **Easy to probe.** An attacker with query access can find the
  vocabulary that trips it. Pair it with Tier-1 structural checks and
  a Tier-3 judge; treat it as a cheap, fast, auditable layer for a
  *specific* failure cluster, not as the whole defence.
