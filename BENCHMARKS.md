# Benchmarks

Generated from the result files in `bench/results/` by `python bench.py --render-benchmarks`. **Do not edit by hand.** Every figure below is read out of one of those JSON files; a number in this document that is not derivable from a shipped result file is a defect.

Reproduce, from a fresh clone:

    ./install.sh && python bench.py --all --corpus bench/corpus --corpus-quality bench/corpus-quality

What produced the numbers below. **This table has no N and no mode of its own — it is where the N and the modes for every other table are declared.**

|  |  |
|---|---|
| source files | `bench/results/2026-07-20-chunk-small.json`, `bench/results/2026-07-20-baseline.json`, `bench/results/2026-07-20-chunk-large.json` |
| run date | 2026-07-20 |
| profiles | `chunk-small`, `baseline`, `chunk-large` |
| samples per operation | create **20**, read **100**, update **50**, delete **100**, cold start **5** |
| modes | `warm` (many operations in one process) and `cold-start` (a fresh process per sample), never mixed in one table |
| corpus (timed) | `bench/corpus/` — 7 files, 50699 characters, seed 20260720, 18 planted facts. **Every timed figure in this document comes from this corpus and nothing else.** |
| corpus (labelled, untimed) | `bench/corpus-quality/` — 24 files, 265979 characters, seed 20260721, 92 planted facts, 24 x 8 planting slots. **No timed operation touches it**; it exists because the labelled set needs more planting slots than the timed corpus has, and growing the timed corpus would move every latency figure above. |
| network | blocked; blocked outbound connects during the run: 0 |

## Environment

The machine every figure below was produced on. **No N and no mode: this is not a measurement**, it is the configuration the measurements were taken in.

| item | value |
|---|---|
| CPU | Apple M2 Pro |
| cores (physical) | 12 |
| cores (logical) | 12 |
| RAM (GiB) | 16.0 |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| architecture | arm64 |
| Python | 3.12.7 |
| chromadb | 1.5.9 |
| onnxruntime | 1.27.0 |
| numpy | 2.5.1 |

All three profiles were measured on the same machine in the same session. No hostname, user name or file path is recorded: the reader needs the hardware, not the workstation it belongs to.

## Method

Four operations are timed against a synthetic corpus generated deterministically from seed 20260720 and shipped in this repository as `bench/corpus/`. The corpus on disk is compared byte-for-byte with the generator's output before anything is measured, and the run fails loudly if they differ — otherwise the numbers would describe a corpus nobody can see.

| operation | N | mode | what is timed |
|---|---|---|---|
| create | 20 | warm | a full index build of the whole corpus |
| read | 100 | warm | one pass over 5 fixed queries at k=5, fixed order, no sampling |
| update | 50 | warm | re-index of a single changed note |
| delete | 100 | warm | removal of one note |
| delete +lag | 100 | warm | time until the removed note stops being returned by a probe query |
| cold start | 5 | cold-start | a fresh subprocess per sample, one operation each |

**Each operation has its own N, and the statistics are gated on it.** The four operations differ by two orders of magnitude in cost, so one N for the harness either starves the cheap operations of samples or makes the expensive ones unrunnable. Median, minimum, maximum and median absolute deviation are reported for every operation: they are statements about the samples in hand. **P50 and P95 are reported only at N ≥ 100** — below that a percentile is an interpolation between two neighbouring samples presented as a property of a distribution. So read (N=100), delete (N=100) carry percentiles; create (N=20), update (N=50) do not, and report a median with a MAD instead.

**There is no P99 in this document, at any N.** Not suppressed — not computed. At N=100 the 99th percentile is an interpolation between the two largest samples, which is a way of printing the maximum and calling it a tail statistic; at N=20 it simply *is* the maximum. No sample count used anywhere in this harness earns one, so the code does not produce one.

**MAD rather than standard deviation at small N.** The median absolute deviation — `median(|x - median(x)|)` — asks how far a typical sample sits from a typical sample. A standard deviation assumes the tail was sampled and is dragged around by the one slow run every process has, which at N=20 is a statement about scheduling luck rather than about the operation. Both are in the result JSONs; the tables show the MAD.

**Two modes, named, never mixed.** `warm` is many operations in one process: the model is loaded, the OS cache is hot, and the marginal cost of one more operation is what is being measured. `cold-start` is a fresh subprocess per sample, which pays interpreter startup, module import, Chroma client construction and the ONNX model load before it does any work. Every table below states which mode produced it. The four timed operations and the stage split are warm; the cold-start section is not.

**Quality is measured, not assumed.** Latency alone ranks a system that answers fast and wrong above one that answers slowly and right, so every run also scores document-level `recall@k` for k in 1, 3, 5, 10, plus MRR and `ndcg@10`, over 8 labelled queries planted into the corpus at generation time. Relevance is document-level: a retrieved chunk counts when its source file is in the query's relevant set. Chunk identity is the independent variable across these profiles, so chunk-keyed labels would make them incomparable by design. Scoring runs after all four timed operations and takes no timing samples.

**Two controls run every time, and are reported with their real numbers.** They exist because a check that can only pass measures nothing.

1. *An unanswerable query.* No document answers it, so it must score zero at every rung. It deliberately reuses vocabulary the corpus is full of, so the retriever answers it confidently with plausible chunks — the control has to survive a confident wrong answer, which is exactly the failure latency cannot see.
2. *Deranged labels.* Every query is scored against another query's documents, with the swap required to be document-disjoint from the truth. The aggregate must drop; if it does not, the labels are not independent of the retriever.

**Read latency is decomposed, and the decomposition has to close.** Query embedding, vector search and result materialization are timed separately on the same query in the same iteration, N=100 passes over the 5 fixed queries. Materialization is a *difference* of two measurements — the same search with and without documents, metadatas and distances — so on an individual sample it can come out negative; those samples are floored at zero and **counted**, and the count is in the table. The three stages are required to sum to within 10% of the full in-process read, and a run whose split does not close fails rather than publishing a decomposition that does not add up.

The full MCP round-trip is measured alongside them, over stdio, at N=20. It is an **outer envelope, not a summand**: it contains a process hop, JSON-RPC framing on both sides and the server's own formatting work on top of everything the in-process stages measure. The `initialize` handshake and one warm-up call happen outside the timed samples, so what is timed is request→response on a warm server. The server is launched against this harness's own collection and its first reply is checked to name a corpus document — a round trip that answered from some other collection is a failed run, not a fast one.

**Retrieval is scored twice: at equal k, and under a token budget.** Equal k is not a fair comparison across chunk geometries — 5 chunks of 239.2 characters and 5 chunks of 1087.0 characters are not the same context. The budget-constrained scoring retrieves the top 128 chunks and then admits them under a fixed budget, which is the constraint a caller actually has.

Two rules of that scoring have to be stated, because a reader cannot check the numbers without them:

1. *Tokens are approximated as `ceil(len(text) / 4)`.* **This is a character heuristic, not a tokenizer.** Real BPE counts move with vocabulary, casing and whitespace, and this approximation will be wrong for any specific model — discount the absolute token figures accordingly. It is applied identically to every profile, so it cannot favour one of them; what it cannot support is a claim about a real model's context window.
2. *Admission is a greedy **prefix**.* Chunks are walked in rank order and admitted while the running total stays within budget; the walk **stops** at the first chunk that would exceed it. It does not skip that chunk and carry on down the ranking. Skip-and-continue is a different retriever and a flattering one: it would let a profile with many small chunks backfill the remaining budget with lower-ranked text while a profile with large chunks got truncated, and the comparison would then be measuring the packer rather than the chunking.

**Offline by design, and checked.** A socket guard is installed before anything heavy is imported; loopback and unix sockets stay allowed, every other outbound connect is refused and counted into the report. The runs below recorded 0, 0, 0 blocked attempts.

## Latency — mode `warm`

Many operations in one process: the model is loaded, the OS cache is hot, and what is measured is the marginal cost of one more operation. This is v1's only mode, named honestly. For the cost of the first operation in a fresh process, see [Cold start](#cold-start).

Median, milliseconds. **Mode `warm`; N per row in the `n` column:**

| operation (median ms) | n | chunk-small | baseline | chunk-large |
|---|---|---|---|---|
| create | 20 | 3729.68 | 1060.96 | 769.07 |
| read | 100 | 82.73 | 75.40 | 73.54 |
| update | 50 | 288.61 | 111.96 | 95.35 |
| delete | 100 | 9.06 | 8.40 | 10.74 |
| delete +lag | 100 | 16.15 | 15.38 | 16.16 |

Full distribution, milliseconds. `P50`/`P95` are blank where N < 100; there is no P99 column because no N here earns one:

| profile | operation | mode | n | median | MAD | min | max | P50 | P95 |
|---|---|---|---|---|---|---|---|---|---|
| chunk-small | create | warm | 20 | 3729.68 | 79.51 | 3636.33 | 4403.51 | - | - |
| chunk-small | read | warm | 100 | 82.73 | 3.34 | 76.39 | 146.69 | 82.73 | 110.20 |
| chunk-small | update | warm | 50 | 288.61 | 29.05 | 203.38 | 320.75 | - | - |
| chunk-small | delete | warm | 100 | 9.06 | 1.18 | 5.90 | 15.32 | 9.06 | 12.62 |
| chunk-small | delete +lag | warm | 100 | 16.15 | 0.27 | 15.41 | 21.54 | 16.15 | 17.52 |
| baseline | create | warm | 20 | 1060.96 | 6.95 | 1046.35 | 1074.66 | - | - |
| baseline | read | warm | 100 | 75.40 | 0.41 | 73.89 | 131.93 | 75.40 | 76.52 |
| baseline | update | warm | 50 | 111.96 | 8.06 | 93.48 | 160.69 | - | - |
| baseline | delete | warm | 100 | 8.40 | 1.83 | 4.49 | 19.03 | 8.40 | 13.40 |
| baseline | delete +lag | warm | 100 | 15.38 | 0.18 | 15.05 | 18.93 | 15.38 | 15.97 |
| chunk-large | create | warm | 20 | 769.07 | 3.51 | 761.15 | 780.03 | - | - |
| chunk-large | read | warm | 100 | 73.54 | 0.52 | 72.01 | 139.49 | 73.54 | 77.00 |
| chunk-large | update | warm | 50 | 95.35 | 6.40 | 81.49 | 142.60 | - | - |
| chunk-large | delete | warm | 100 | 10.74 | 1.42 | 5.71 | 19.00 | 10.74 | 14.91 |
| chunk-large | delete +lag | warm | 100 | 16.16 | 0.36 | 15.37 | 28.27 | 16.16 | 21.55 |

Corpus as each profile cut it. **No N and no mode: this is not a sampled measurement.** It is a deterministic property of the chunker applied to a fixed corpus — running it again produces the same numbers, which is why there is nothing to take a median of:

| profile | max chars | overlap | files | chunks | mean chunk | largest chunk |
|---|---|---|---|---|---|---|
| chunk-small | 300 | 40 | 7 | 256 | 239.2 | 310 |
| baseline | 1200 | 150 | 7 | 66 | 823.8 | 1211 |
| chunk-large | 2000 | 250 | 7 | 48 | 1087.0 | 1997 |

## Cold start

**mode `cold-start`, n=5 per cell.** A fresh subprocess per sample, invoked directly so the interpreter starts exactly once. Each sample pays module import, Chroma client construction and the ONNX model load before it does any work. `process wall` is measured by the parent around the whole subprocess and is the headline; `init` and `operation` are the child's own split of it; `startup/teardown` is what is left over — `wall − (init + operation)` — reported rather than absorbed into either neighbour.

**Median, minimum and maximum only.** N=5 earns no percentile of any kind, and none is shown.

`read` — one pass over the 5 fixed queries:

| profile | component | mode | n | median | min | max |
|---|---|---|---|---|---|---|
| chunk-small | process wall (total) | cold-start | 5 | 674.27 | 667.00 | 695.49 |
| chunk-small | init | cold-start | 5 | 478.40 | 472.83 | 492.08 |
| chunk-small | operation | cold-start | 5 | 88.03 | 79.53 | 88.57 |
| chunk-small | startup/teardown | cold-start | 5 | 110.66 | 109.14 | 115.38 |
| baseline | process wall (total) | cold-start | 5 | 672.43 | 663.26 | 691.06 |
| baseline | init | cold-start | 5 | 477.67 | 465.61 | 494.57 |
| baseline | operation | cold-start | 5 | 74.86 | 74.31 | 76.17 |
| baseline | startup/teardown | cold-start | 5 | 117.67 | 116.67 | 130.65 |
| chunk-large | process wall (total) | cold-start | 5 | 640.05 | 635.56 | 651.28 |
| chunk-large | init | cold-start | 5 | 448.06 | 446.47 | 453.15 |
| chunk-large | operation | cold-start | 5 | 76.42 | 75.08 | 84.69 |
| chunk-large | startup/teardown | cold-start | 5 | 114.96 | 112.42 | 119.06 |

`create` — a full index build, in a fresh process:

| profile | component | mode | n | median | min | max |
|---|---|---|---|---|---|---|
| chunk-small | process wall (total) | cold-start | 5 | 4089.47 | 4085.49 | 4113.31 |
| chunk-small | init | cold-start | 5 | 441.67 | 434.53 | 444.88 |
| chunk-small | operation | cold-start | 5 | 3527.03 | 3523.67 | 3561.87 |
| chunk-small | startup/teardown | cold-start | 5 | 120.72 | 116.42 | 123.85 |
| baseline | process wall (total) | cold-start | 5 | 1545.22 | 1527.79 | 1588.60 |
| baseline | init | cold-start | 5 | 442.26 | 435.25 | 447.66 |
| baseline | operation | cold-start | 5 | 967.86 | 956.52 | 987.91 |
| baseline | startup/teardown | cold-start | 5 | 133.97 | 129.70 | 160.90 |
| chunk-large | process wall (total) | cold-start | 5 | 1285.61 | 1281.32 | 1301.72 |
| chunk-large | init | cold-start | 5 | 446.06 | 440.41 | 453.07 |
| chunk-large | operation | cold-start | 5 | 713.71 | 709.60 | 728.08 |
| chunk-large | startup/teardown | cold-start | 5 | 124.48 | 122.40 | 133.23 |

On `chunk-small` the same read pass costs 82.73 ms warm and 674.27 ms from a fresh process — a factor of **8x**, essentially all of it paid before the first query is issued. v1 reported a `cold` column that was the first sample of a warm loop, in a process that had already imported everything and loaded the model; that column is gone.

**This is a cold *process*, not a cold *machine*.** The OS page cache still holds the ONNX model file after the first sample, and it is not purged between samples — purging it needs privileges this harness does not take. The first sample of the first profile pays a genuinely cold file read; the rest do not. Treat these figures as the cost of starting a new process on a machine that has run this before, which is the common case for a CLI, and not as a first-boot number.

## Where the read time goes

**mode `warm`, n=500 per profile** (100 passes over 5 fixed queries). Milliseconds per query.

| profile | stage | n | median | MAD | min | max | in the sum? |
|---|---|---|---|---|---|---|---|
| chunk-small | query embedding | 500 | 14.263 | 0.137 | 13.869 | 18.830 | yes |
| chunk-small | vector search | 500 | 1.325 | 0.195 | 0.825 | 5.212 | yes |
| chunk-small | result materialization | 500 | 0.051 | 0.034 | 0.000 | 2.759 | yes |
| chunk-small | **sum of the three** |  | 15.639 |  |  |  |  |
| chunk-small | **full in-process read** | 500 | 15.751 | 0.276 | 14.945 | 23.180 | is the total |
| chunk-small | full MCP round-trip | 20 | 83.150 | 2.131 | 79.729 | 91.261 | **no — envelope** |
| baseline | query embedding | 500 | 14.265 | 0.102 | 13.918 | 18.501 | yes |
| baseline | vector search | 500 | 0.370 | 0.027 | 0.315 | 1.221 | yes |
| baseline | result materialization | 500 | 0.087 | 0.021 | 0.000 | 1.483 | yes |
| baseline | **sum of the three** |  | 14.722 |  |  |  |  |
| baseline | **full in-process read** | 500 | 14.805 | 0.125 | 14.356 | 17.505 | is the total |
| baseline | full MCP round-trip | 20 | 79.528 | 1.038 | 78.388 | 92.668 | **no — envelope** |
| chunk-large | query embedding | 500 | 14.654 | 0.160 | 14.148 | 19.637 | yes |
| chunk-large | vector search | 500 | 0.418 | 0.025 | 0.348 | 4.182 | yes |
| chunk-large | result materialization | 500 | 0.061 | 0.028 | 0.000 | 5.175 | yes |
| chunk-large | **sum of the three** |  | 15.133 |  |  |  |  |
| chunk-large | **full in-process read** | 500 | 15.284 | 0.213 | 14.739 | 21.633 | is the total |
| chunk-large | full MCP round-trip | 20 | 81.264 | 2.475 | 78.336 | 92.963 | **no — envelope** |

How well the decomposition closes, and what had to be floored. Same samples as the table above — **mode `warm`, n=500 per profile** — restated so the residual is not read against a different N:

| profile | sum of stages | full read | residual | limit | materialization samples floored at 0 |
|---|---|---|---|---|---|
| chunk-small | 15.639 | 15.751 | 0.71% | 10% | 65 / 500 (13.0%) |
| baseline | 14.722 | 14.805 | 0.56% | 10% | 22 / 500 (4.4%) |
| chunk-large | 15.133 | 15.284 | 0.99% | 10% | 53 / 500 (10.6%) |

The residual is interpreter and call overhead that belongs to no stage. It is inside the 10% limit on every profile; a run where it is not fails and publishes nothing. The floored samples are the ones where materialization — the difference between the same search with and without its payload — came out negative through measurement noise; they are clamped to zero and counted here rather than silently.

**Do not read the materialization column as a measurement on `chunk-small` or `chunk-large`.** 13.0% on `chunk-small` and 10.6% on `chunk-large` of its samples floored, which means the stage is smaller than the run-to-run noise of the two searches it is the difference of. Its median is a lower bound on a quantity this clock cannot resolve, not an estimate of it. The number is left in the table because deleting it would make the three stages appear to sum more cleanly than they do; the two stages above it, and the total, are measurements.

The MCP round-trip is the envelope, not a summand: it adds a process hop, JSON-RPC framing on both sides and the server's own result formatting on top of everything above it. It is measured at N=20 against the same collection, on a server that has already answered one untimed query.

## Retrieval quality

Equal k — every profile retrieves the same *number of chunks*. **N = 8 labelled queries over 7 documents, retrieval depth 10 chunks.** **No mode applies: this is untimed.** Scoring runs after all four timed operations and takes no timing samples, so there is no warm or cold-start reading of it to give — the absence of a mode column here is a property of the measurement, not an omission.

| profile | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|
| chunk-small | 0.396 | 0.938 | 1.000 | 1.000 | 0.917 | 0.946 |
| baseline | 0.458 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chunk-large | 0.396 | 0.708 | 0.875 | 1.000 | 0.893 | 0.900 |

`recall@1` cannot reach 1.0: with 2 to 3 relevant documents per query (mean 2.25), one retrieved chunk can cover at most one of them, which caps the mean at **0.458**.

### Controls

**Also untimed, so again no mode applies.** The two controls carry different N and it matters which is which: the deranged-label control re-scores the same **8 labelled queries** against document-disjoint label sets, so its columns are means over 8 queries; the unanswerable control is **a single query**, deliberately not one of the 8, and its columns are that one query's own scores rather than a mean.

| profile | unanswerable rr | unanswerable ndcg@10 | docs it still returned | deranged label overlap | MRR true → deranged | recall@3 true → deranged |
|---|---|---|---|---|---|---|
| chunk-small | 0.000 | 0.000 | 5 | 0 | 0.917 → 0.351 | 0.938 → 0.312 |
| baseline | 0.000 | 0.000 | 6 | 0 | 1.000 → 0.266 | 1.000 → 0.250 |
| chunk-large | 0.000 | 0.000 | 6 | 0 | 0.893 → 0.308 | 0.708 → 0.354 |

The unanswerable query scored zero on every profile while the retriever confidently returned several distinct documents for it. Deranged labels are document-disjoint from the truth — overlap 0 — and every profile dropped.

## Retrieval quality under a token budget

The same labelled queries, scored again with the *context* held fixed instead of the chunk count. Untimed, run after every timed operation. Means over the 8 labelled queries. Tokens are `ceil(len(text) / 4)` — an approximation, see Method.

Budget **2000 tokens** — **untimed, so no mode applies; means over 8 labelled queries:**

| profile | recall@budget | unique docs | chunks admitted | tokens used | duplicate source rate | relevant token fraction |
|---|---|---|---|---|---|---|
| chunk-small | 1.000 | 6.88 | 32.50 | 1962.8 | 0.788 | 0.393 |
| baseline | 1.000 | 5.50 | 8.75 | 1942.5 | 0.361 | 0.492 |
| chunk-large | 0.938 | 4.25 | 6.50 | 1796.5 | 0.307 | 0.533 |

Budget **4000 tokens** — **untimed, so no mode applies; means over 8 labelled queries:**

| profile | recall@budget | unique docs | chunks admitted | tokens used | duplicate source rate | relevant token fraction |
|---|---|---|---|---|---|---|
| chunk-small | 1.000 | 7.00 | 66.12 | 3960.9 | 0.894 | 0.331 |
| baseline | 1.000 | 6.50 | 18.62 | 3880.5 | 0.647 | 0.390 |
| chunk-large | 1.000 | 6.00 | 15.62 | 3880.8 | 0.608 | 0.347 |

The unanswerable control was run under every budget on every profile and scored `recall@budget` 0 in all of them; a non-zero score there fails the run, exactly as it does at equal k.

## Retrieval quality — the labelled set

Phase 1 scored retrieval over eight queries on the timed corpus. One miss there is a 12.5 point swing, and `recall@budget` came out saturated at 1.000 on two of the three profiles, so the comparison resolved the bottom of the field and not the top. This section is the replacement: **64 labelled queries** (56 answerable, 8 unanswerable controls) over a **second, larger generated corpus** of 24 documents and 265979 characters, seed 20260721. **The timed corpus is untouched by any of it** — every latency figure above is a property of `bench/corpus`, which this section never reads.

Every query carries a **category**, a **split** and a **phrasing**. **Untimed throughout, so no mode applies to any table in this section.** The counts, which are read out of the shipped ground-truth file rather than typed here:

| category | queries | held-out | tuning | relevant documents |
|---|---|---|---|---|
| multi-doc | 20 | 15 | 5 | 2 or 3, and each carries a different part of the answer |
| single-hop | 24 | 18 | 6 | exactly 1 relevant document |
| unanswerable | 8 | 6 | 2 | 0; must score exactly zero on every rung and budget |
| vague | 12 | 9 | 3 | 1 or 2, underspecified phrasing, still answerable |

Phrasing is balanced across the answerable set — 16 misspelled, 16 natural, 16 terse, 16 verbose — and covers how people actually ask: a full question (`natural`), a keyword fragment with no sentence structure (`terse`), a conversational ask with filler the retriever has to see past (`verbose`), and one plausible typo in a content word, never in the invented entity noun (`misspelled`). A typo in the invented noun would test exact match and nothing else.

### The headline: held-out queries only

The three profiles already existed and were selected while the phase-1 eight-query set was visible. A labelled set that is entirely visible to profile selection cannot support a claim about profile selection, so the set carries a declared split and **the headline is the held-out subset**. The rule, as recorded in the shipped ground-truth file: queries ordered by category, then by phrasing rotated by the category's own position, then by id; every 4th entry in that order is tuning and the rest are held-out. Deterministic, exactly a quarter of every category and exactly a quarter of every phrasing, and stable across a rebuild - a split that moved would invalidate every comparison against an earlier run.

**N = 42 held-out answerable queries. Untimed, so no mode applies.**

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 42 | 0.556 | 0.778 | 0.829 | 0.865 | 0.833 | 0.811 |
| baseline | 42 | 0.345 | 0.544 | 0.683 | 0.766 | 0.667 | 0.626 |
| chunk-large | 42 | 0.139 | 0.238 | 0.313 | 0.472 | 0.354 | 0.337 |

`recall@1` is a live discriminator here and was not one in phase 1. Every phase-1 query had two or three relevant documents, which caps the mean `recall@1` by construction; 24 of the queries in this set have **exactly one** relevant document, so on that subset one retrieved chunk can reach 1.000.

### Held-out against tuning

Both are reported; only the first is the headline. If the two disagree about the ranking, that disagreement is itself a finding and is stated below the table. **Untimed; each row states its own N.**

Split **held-out** — n = 42 answerable queries:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 42 | 0.556 | 0.778 | 0.829 | 0.865 | 0.833 | 0.811 |
| baseline | 42 | 0.345 | 0.544 | 0.683 | 0.766 | 0.667 | 0.626 |
| chunk-large | 42 | 0.139 | 0.238 | 0.313 | 0.472 | 0.354 | 0.337 |

Split **tuning** — n = 14 answerable queries:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 14 | 0.679 | 0.845 | 0.845 | 0.893 | 0.929 | 0.883 |
| baseline | 14 | 0.321 | 0.583 | 0.786 | 0.857 | 0.639 | 0.647 |
| chunk-large | 14 | 0.238 | 0.262 | 0.357 | 0.548 | 0.369 | 0.383 |

The two splits agree on the ordering by `recall@3`: `chunk-small` > `baseline` > `chunk-large`. That agreement is worth stating rather than assuming — it is the reason the held-out number can be read as the profile's behaviour and not as an artefact of which queries landed in which subset.

### By category, held out

What *kind* of query separates the profiles, if any does. **Untimed; every row states its own N, and a row computed from fewer than 8 queries is marked † and is indicative only — it is printed so the shape is visible, not so it can be quoted.**

Category **multi-doc**, held out — n = 15:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 15 | 0.422 | 0.844 | 0.889 | 0.889 | 1.000 | 0.903 |
| baseline | 15 | 0.367 | 0.556 | 0.644 | 0.711 | 0.922 | 0.706 |
| chunk-large | 15 | 0.222 | 0.333 | 0.444 | 0.556 | 0.640 | 0.495 |

Category **single-hop**, held out — n = 18:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 18 | 0.778 | 0.889 | 0.944 | 0.944 | 0.844 | 0.869 |
| baseline | 18 | 0.333 | 0.556 | 0.722 | 0.722 | 0.481 | 0.540 |
| chunk-large | 18 | 0.056 | 0.111 | 0.111 | 0.278 | 0.096 | 0.144 |

Category **vague**, held out — n = 9:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 9 | 0.333 | 0.444 | 0.500 | 0.667 | 0.532 | 0.542 |
| baseline | 9 | 0.333 | 0.500 | 0.667 | 0.944 | 0.613 | 0.665 |
| chunk-large | 9 | 0.167 | 0.333 | 0.500 | 0.722 | 0.392 | 0.460 |

### By phrasing

Both splits, answerable queries only. **Untimed; each row states its own N.** This is the one table in the document that says anything about *how the question was asked* rather than about the index.

Phrasing **misspelled** — n = 14:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 14 | 0.583 | 0.726 | 0.750 | 0.833 | 0.866 | 0.810 |
| baseline | 14 | 0.298 | 0.440 | 0.607 | 0.714 | 0.657 | 0.593 |
| chunk-large | 14 | 0.107 | 0.262 | 0.357 | 0.476 | 0.417 | 0.370 |

Phrasing **natural** — n = 14:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 14 | 0.679 | 0.821 | 0.821 | 0.821 | 0.857 | 0.830 |
| baseline | 14 | 0.357 | 0.607 | 0.821 | 0.893 | 0.651 | 0.689 |
| chunk-large | 14 | 0.179 | 0.214 | 0.321 | 0.500 | 0.335 | 0.342 |

Phrasing **terse** — n = 14:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 14 | 0.476 | 0.738 | 0.869 | 0.869 | 0.836 | 0.816 |
| baseline | 14 | 0.274 | 0.488 | 0.619 | 0.690 | 0.627 | 0.577 |
| chunk-large | 14 | 0.119 | 0.214 | 0.298 | 0.452 | 0.307 | 0.305 |

Phrasing **verbose** — n = 14:

| profile | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| chunk-small | 14 | 0.607 | 0.893 | 0.893 | 0.964 | 0.869 | 0.862 |
| baseline | 14 | 0.429 | 0.679 | 0.786 | 0.857 | 0.704 | 0.668 |
| chunk-large | 14 | 0.250 | 0.286 | 0.321 | 0.536 | 0.373 | 0.378 |

### Are these scores reproducible?

The timed corpus is small enough that its retrieval scores come out identical run after run. This one is not, and rather than round the question away it is measured: the same corpus indexed a second time into a separate collection, the same labelled queries re-issued against it, and the two ranked lists compared position by position. **Untimed; n = 64 labelled queries per profile.**

| profile | queries | identical rankings | differing rankings | queries whose recall changed | largest headline shift |
|---|---|---|---|---|---|
| chunk-small | 64 | 45 | 19 | 9 | 0.039682 |
| baseline | 64 | 58 | 6 | 0 | 0.001190 |
| chunk-large | 64 | 61 | 3 | 2 | 0.023810 |

Some ranked lists are **not** reproduced exactly: 28 across the three profiles. The cause is the vector store's approximate index, which this corpus is large enough to engage and the timed corpus is not — two builds can order near-ties differently. What it costs is measured rather than assumed: the largest shift in any held-out headline metric on any profile is **0.039682**. Read every labelled figure in this document as carrying at least that much reproduction noise, and treat a gap of that size as no gap.

At least, and not exactly: this is **one rebuild inside one run on one machine**, so it is a lower bound on run-to-run variation and not a ceiling. A separate run of the same code over the same committed corpora is free to move a labelled figure further than this. The timed corpus's own equal-k quality block, by contrast, does reproduce exactly — it is small enough that the search is not approximate at all.

### Controls on the labelled set

**Untimed, so again no mode applies.** Two controls, and both are enforced rather than reported. A non-zero unanswerable score raises before anything is written at all. A deranged-label set that fails to drop, or that overlaps the truth, sets a non-zero exit the way a state-guard problem does — the result file for that profile exists, and this document is not regenerated from it.

| profile | unanswerable queries | their scores | deranged label overlap | derangement strategy | recall@3 true → deranged |
|---|---|---|---|---|---|
| chunk-small | 8 | 0.000 at every rung | 0 | global | 0.795 → 0.071 |
| baseline | 8 | 0.000 at every rung | 0 | global | 0.554 → 0.074 |
| chunk-large | 8 | 0.000 at every rung | 0 | global | 0.244 → 0.092 |

The 8 unanswerable queries are built the same way the phase-1 control was: a family noun the corpus is full of, paired with a device noun that occurs nowhere in it. The retriever answers them confidently with lexically plausible chunks, and every rung, `rr` and `nDCG` must still be exactly zero. The derangement search is seeded and attempt-capped rather than unbounded, and the strategy it used is reported above instead of being assumed.

## The labelled set under a token budget

The same 64 labelled queries, scored again with the *context* held fixed instead of the chunk count. **Untimed; means over held-out answerable queries; tokens are `ceil(len(text) / 4)`, an approximation, see Method.**

Budget **2000 tokens**, split **held-out** — **untimed, n = 42 queries:**

| profile | n | recall@budget | unique docs | chunks admitted | tokens used | duplicate source rate | relevant token fraction |
|---|---|---|---|---|---|---|---|
| chunk-small | 42 | 0.940 | 17.71 | 31.67 | 1970.6 | 0.439 | 0.103 |
| baseline | 42 | 0.758 | 7.71 | 9.21 | 1874.2 | 0.139 | 0.209 |
| chunk-large | 42 | 0.417 | 8.05 | 9.67 | 1877.1 | 0.134 | 0.127 |

Budget 2000, category **multi-doc**, held out — n = 15:

| profile | n | recall@budget | unique docs | relevant token fraction |
|---|---|---|---|---|
| chunk-small | 15 | 1.000 | 17.00 | 0.181 |
| baseline | 15 | 0.689 | 6.53 | 0.316 |
| chunk-large | 15 | 0.500 | 7.33 | 0.224 |

Budget 2000, category **single-hop**, held out — n = 18:

| profile | n | recall@budget | unique docs | relevant token fraction |
|---|---|---|---|---|
| chunk-small | 18 | 0.944 | 18.22 | 0.048 |
| baseline | 18 | 0.722 | 6.94 | 0.113 |
| chunk-large | 18 | 0.167 | 6.61 | 0.022 |

Budget 2000, category **vague**, held out — n = 9:

| profile | n | recall@budget | unique docs | relevant token fraction |
|---|---|---|---|---|
| chunk-small | 9 | 0.833 | 17.89 | 0.083 |
| baseline | 9 | 0.944 | 11.22 | 0.225 |
| chunk-large | 9 | 0.778 | 12.11 | 0.173 |

Budget **4000 tokens**, split **held-out** — **untimed, n = 42 queries:**

| profile | n | recall@budget | unique docs | chunks admitted | tokens used | duplicate source rate | relevant token fraction |
|---|---|---|---|---|---|---|---|
| chunk-small | 42 | 0.976 | 22.19 | 64.57 | 3969.8 | 0.656 | 0.078 |
| baseline | 42 | 0.853 | 14.07 | 19.45 | 3885.2 | 0.248 | 0.130 |
| chunk-large | 42 | 0.639 | 13.79 | 19.95 | 3856.5 | 0.273 | 0.104 |

Budget 4000, category **multi-doc**, held out — n = 15:

| profile | n | recall@budget | unique docs | relevant token fraction |
|---|---|---|---|---|
| chunk-small | 15 | 1.000 | 21.87 | 0.137 |
| baseline | 15 | 0.822 | 13.20 | 0.197 |
| chunk-large | 15 | 0.722 | 13.07 | 0.184 |

Budget 4000, category **single-hop**, held out — n = 18:

| profile | n | recall@budget | unique docs | relevant token fraction |
|---|---|---|---|---|
| chunk-small | 18 | 0.944 | 22.39 | 0.037 |
| baseline | 18 | 0.833 | 13.39 | 0.068 |
| chunk-large | 18 | 0.444 | 12.50 | 0.025 |

Budget 4000, category **vague**, held out — n = 9:

| profile | n | recall@budget | unique docs | relevant token fraction |
|---|---|---|---|---|
| chunk-small | 9 | 1.000 | 22.33 | 0.061 |
| baseline | 9 | 0.944 | 16.89 | 0.145 |
| chunk-large | 9 | 0.889 | 17.56 | 0.128 |

All 8 unanswerable controls were run under every budget on every profile and scored `recall@budget` 0 in all of them; a non-zero score there fails the run, exactly as it does at equal k.

## Multilingual probe

**This is not a profile comparison and is never aggregated into one.** It measures the vocabulary coverage of the embedder this toolkit ships (all-MiniLM-L6-v2, an English model with an English WordPiece vocabulary), not chunk geometry. **Untimed; n = 12 base queries × 5 variants = 60 scored queries per profile**, all drawn from the held-out `single-hop` and `multi-doc` sets.

The external review this v2 answers asked for an EN+DE+RU+UK labelled mix. The embedder this toolkit ships is `all-MiniLM-L6-v2`, an English model with an English WordPiece vocabulary, so a recall number over Cyrillic queries measures that vocabulary and not chunk geometry — which is why the labelled set above is English only and this is a probe. The fourth variant is what makes it evidence rather than an assertion: `ru-latin-anchor` is the same Russian sentence as `ru-cyrillic` with the invented entity nouns left in Latin script, so the two differ in exactly one thing.

Profile `chunk-small` — **untimed, n = 12 queries per variant:**

| variant | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| en-reference | 12 | 0.764 | 0.931 | 0.931 | 0.931 | 1.000 | 0.948 |
| de | 12 | 0.319 | 0.514 | 0.514 | 0.514 | 0.486 | 0.459 |
| ru-cyrillic | 12 | 0.000 | 0.181 | 0.181 | 0.208 | 0.125 | 0.141 |
| uk-cyrillic | 12 | 0.000 | 0.042 | 0.139 | 0.306 | 0.088 | 0.146 |
| ru-latin-anchor | 12 | 0.181 | 0.333 | 0.444 | 0.486 | 0.392 | 0.368 |

Profile `baseline` — **untimed, n = 12 queries per variant:**

| variant | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| en-reference | 12 | 0.486 | 0.597 | 0.750 | 0.806 | 0.767 | 0.703 |
| de | 12 | 0.361 | 0.444 | 0.514 | 0.694 | 0.588 | 0.556 |
| ru-cyrillic | 12 | 0.000 | 0.083 | 0.236 | 0.556 | 0.121 | 0.231 |
| uk-cyrillic | 12 | 0.000 | 0.083 | 0.236 | 0.458 | 0.108 | 0.192 |
| ru-latin-anchor | 12 | 0.069 | 0.264 | 0.458 | 0.556 | 0.302 | 0.333 |

Profile `chunk-large` — **untimed, n = 12 queries per variant:**

| variant | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|---|
| en-reference | 12 | 0.042 | 0.069 | 0.153 | 0.347 | 0.192 | 0.184 |
| de | 12 | 0.042 | 0.153 | 0.222 | 0.375 | 0.204 | 0.223 |
| ru-cyrillic | 12 | 0.000 | 0.250 | 0.361 | 0.556 | 0.156 | 0.259 |
| uk-cyrillic | 12 | 0.000 | 0.083 | 0.319 | 0.639 | 0.130 | 0.258 |
| ru-latin-anchor | 12 | 0.125 | 0.153 | 0.264 | 0.458 | 0.273 | 0.280 |

### What the probe found

The comparison the probe exists for, **per profile rather than averaged**: a mean over three profiles would hide a profile that went the other way, and one of the three things this document is for is not doing that. `anchor − cyrillic` is the cost of transliterating the invented noun and nothing else — the Russian sentence around it is identical.

| profile | recall@3 en | recall@3 anchor | recall@3 cyrillic | recall@3 anchor − cyrillic | mrr en | mrr anchor | mrr cyrillic | mrr anchor − cyrillic |
|---|---|---|---|---|---|---|---|---|
| chunk-small | 0.931 | 0.333 | 0.181 | 0.153 | 1.000 | 0.392 | 0.125 | 0.267 |
| baseline | 0.597 | 0.264 | 0.083 | 0.181 | 0.767 | 0.302 | 0.121 | 0.181 |
| chunk-large | 0.069 | 0.153 | 0.250 | -0.097 | 0.192 | 0.273 | 0.156 | 0.116 |

**`chunk-large` cannot resolve this comparison and is excluded from the verdict below.** Its own English reference scores 0.069 `recall@3` on these twelve questions, which is at the floor: a profile that cannot answer the question in English tells you nothing about what transliterating the noun costs. That is a fact about `chunk-large` retrieval on this corpus — see the labelled-set tables above — not about the probe.

**The Latin entity noun is doing the work, not the question.** On every profile that can resolve it (`chunk-small`, `baseline`), and on both metrics, the same Russian sentence scores higher with the invented noun left in Latin than with it transliterated: recall@3 on `baseline` +0.181; recall@3 on `chunk-small` +0.153; mrr on `baseline` +0.181; mrr on `chunk-small` +0.267. The model is matching a Latin proper noun as an opaque token rather than reading the question, and a Cyrillic query loses that anchor.

What this does **not** say: nothing here is a statement about chunking, and no number in this section enters any profile ranking, any `by_category` mean or any headline. Swapping in a multilingual embedder is a different comparison and is not in this run.

## Findings

### 1. Read latency is query embedding, not index size

v1 claimed that on a small index the vector search is nearly free relative to MiniLM inference, and inferred it from the fact that read latency barely moved when the index size did. v2 measures it: on this corpus, **query embedding is 91–96% of the full in-process read** across the three profiles. The claim is now backed by the decomposition rather than by an absence of variation. Medians, **mode `warm`, n=500 per profile** — the same samples as [Where the read time goes](#where-the-read-time-goes), restated here:

| profile | chunks in index | query embedding (ms) | vector search (ms) | materialization (ms) | embedding share of the read |
|---|---|---|---|---|---|
| chunk-small | 256 | 14.263 | 1.325 | 0.051 | **90.5%** |
| baseline | 66 | 14.265 | 0.370 | 0.087 | **96.3%** |
| chunk-large | 48 | 14.654 | 0.418 | 0.061 | **95.9%** |

The vector search itself costs 0.370–1.325 ms while the number of chunks in the index spans **5.3x** (48 → 256). The embedding is a fixed cost per query that the chunking cannot move, and at this corpus size it dominates everything the index does. That is what makes read latency look flat: the median read pass spans only **1.13x** across the three profiles (73.54 ms on `chunk-large`, 82.73 ms on `chunk-small`), against a 6.7x span in the chunk budget and a 4.5x span in the mean chunk actually produced (239.2 → 1087.0 characters).

**The scope of that finding is this corpus.** 48 to 256 chunks is small enough that the ANN structure is doing almost nothing; the sentence above is about a small index, and nothing here measures where the crossover is.

### 2. Timing alone picks the wrong profile

On this corpus the timed operations do not separate the profiles on the axis a reader would care about. `chunk-large` is both the fastest to read and the cheapest to build, and if the tables stopped at latency it would look like the answer.

**Quality is what decides it.** At equal k, `baseline` leads on `recall@3` (1.000) and on `ndcg@10` (1.000), and it is the profile this toolkit ships as its default. The full ladder — note that this table deliberately puts three different N side by side: the latency columns are **mode `warm`**, read at **n=100** and create at **n=20**, while the quality columns are **untimed** means over **8 labelled queries**:

| profile | max chars | read median | create median | recall@3 | ndcg@10 |
|---|---|---|---|---|---|
| chunk-small | 300 | 82.73 | 3729.68 | 0.938 | 0.946 |
| baseline | 1200 | 75.40 | 1060.96 | 1.000 | 1.000 |
| chunk-large | 2000 | 73.54 | 769.07 | 0.708 | 0.900 |

Read the table as a decision, not as a ranking: the coarsest profile buys its speed with retrieval quality, the finest one costs more to build *and* scores lower, and the difference is invisible to every timed operation above.

### 3. A token budget does not reorder the profiles, but it changes how far apart they are

The comparison below is `recall@3` at equal k against `recall@budget` at 2000 tokens. **Both are untimed means over the same 8 labelled queries, so no mode applies and the two columns share an N** — that is what makes them comparable at all. Ties are shown as ties: no secondary column is used to break them, because a tie-break would invent an ordering the metric does not support — and two profiles that equal k separates and a budget does not is precisely the outcome this comparison exists to detect.

| profile | recall@3 (equal k=5) | recall@budget (2000 tok) | delta | chunks admitted | unique docs | duplicate source rate |
|---|---|---|---|---|---|---|
| chunk-small | 0.9375 | 1.0000 | 0.0625 | 32.50 | 6.88 | 0.788 |
| baseline | 1.0000 | 1.0000 | 0.0000 | 8.75 | 5.50 | 0.361 |
| chunk-large | 0.7083 | 0.9375 | 0.2292 | 6.50 | 4.25 | 0.307 |

The same two columns as a ranking — **still untimed, still the same 8 labelled queries:**

| order | at equal k | under a 2000-token budget |
|---|---|---|
| ranking | `baseline` > `chunk-small` > `chunk-large` | `chunk-small` = `baseline` > `chunk-large` |

**No profile overtakes another — but the gap that equal k reports is not there once the budget is the constraint.** `baseline` leads `chunk-small` by 0.0625 at equal k (1.0000 vs 0.9375) and ties it under a 2000-token budget (1.0000 vs 1.0000). The ordering survives; the separation does not, and the separation is what the equal-k table was being read for.

**v1's equal-k comparison is the reason this was invisible.** k is a count of chunks, so at k=5 a coarse profile is handed several times as much text as a fine one for the same nominal retrieval depth, and then credited for covering more documents with it. Hold the *context* fixed instead and that advantage is gone: inside 2000 tokens `chunk-small` admits 32.5 chunks covering 6.88 of the 7 documents; `baseline` admits 8.8 chunks covering 5.50 of the 7 documents; `chunk-large` admits 6.5 chunks covering 4.25 of the 7 documents.

One caveat on this table, which the numbers make unavoidable: `recall@budget` is **saturated at 1.000 on `chunk-small` and `baseline`** at this budget. A metric pinned at its ceiling cannot rank anything above it, so the comparison above resolves the bottom of the field and not the top. The corpus has 7 documents and 8 labelled queries; separating profiles that all retrieve everything asked of them needs a harder labelled set, not a different budget.

### 4. The larger held-out set reverses the phase-1 ranking

Phase 1 could not answer this: eight labelled queries, one miss worth 12.5 points, and `recall@budget` saturated at 1.000 on both `baseline` and `chunk-small`. The comparison below is over **42 held-out answerable queries on the quality corpus, untimed**, and a gap is called a separation only when it clears **0.0397** — the larger of what one query can move the mean at this N (0.0238) and what rebuilding the index moves a headline metric by (0.0397, measured above). Both are computed, neither is chosen.

| metric | `baseline` | `chunk-small` | difference | verdict |
|---|---|---|---|---|
| recall@1 | 0.3452 | 0.5556 | -0.2103 | separated |
| recall@3 | 0.5437 | 0.7778 | -0.2341 | separated |
| recall@5 | 0.6825 | 0.8294 | -0.1468 | separated |
| recall@10 | 0.7659 | 0.8651 | -0.0992 | separated |
| mrr | 0.6667 | 0.8331 | -0.1665 | separated |
| ndcg@10 | 0.6264 | 0.8114 | -0.1850 | separated |

**The ranking reverses.** On the timed corpus's eight-query set, `baseline` > `chunk-small` by `recall@3` (1.0000 against 0.9375). On 42 held-out queries over the larger corpus the order is the other way (0.5437 against 0.7778). Two labelled sets over two corpora disagree about which of these profiles is better, and that is the finding: at eight queries the phase-1 ordering was not a property of the profiles.

Two things changed at once and the reversal cannot be attributed to either alone: the labelled set grew from 8 queries to 42 held-out ones, **and** it moved to a different corpus with more notes and longer ones. What can be said is narrower and still worth saying — the phase-1 ordering survived neither change, so it was not robust enough to choose a profile on. Separating the two causes needs this labelled set run against the timed corpus, and its 36 planting slots cannot hold it.

Whatever the branch above says, it is a statement about **42 held-out queries on one synthetic corpus, untimed**. It is not a statement about latency, about a real knowledge base, or about any corpus larger than this one.

## Limitations

These numbers are worth exactly what their scope allows, and the scope is small.

- **One machine, one session.** Every figure comes from the single environment in the table above. Nothing here is a cross-machine claim, and the absolute milliseconds will not transfer.
- **A small synthetic corpus.** 7 documents, 50699 characters, 8 labelled queries with 2 to 3 relevant documents each. That resolves the *direction* of the quality difference; it does not pin its magnitude, and a handful of queries carry the coarse profile's deficit.
- **Synthetic, not natural, text.** The corpus is generated so that it can be shipped and re-derived byte-for-byte. Real notes have different heading density and different vocabulary overlap, both of which the chunker is sensitive to.
- **Local only, single client, no concurrency.** One process, one collection, no parallel readers or writers. The MCP round-trip is the only figure here that crosses a process boundary, and it is a single client talking to a single server. Nothing here says anything about behaviour under load.
- **One embedder.** chroma-onnx / all-MiniLM-L6-v2 throughout. The comparison is between chunk geometries at a fixed embedder, not between embedders. Since query embedding turns out to be most of the read, that fixed choice is doing more work in these numbers than any chunking decision.
- **Cold start is a cold process, not a cold machine.** Each cold-start sample is a genuinely fresh subprocess — new interpreter, new import, new Chroma client, new ONNX session — but the OS page cache still holds the model file from the previous sample, and this harness does not purge it. Only the very first sample pays a cold file read. A first-boot number would be larger, and nothing here measures it.
- **5 cold-start samples per cell.** The cold-start tables report median, minimum and maximum and nothing else, because that many samples support nothing else. The spread between minimum and maximum is the honest statement of what is known about that distribution.
- **Token counts are approximated from characters.** `ceil(len(text) / 4)`, not a tokenizer. The budget comparison is internally consistent and the ranking it produces is not sensitive to a uniform scaling of that estimate, but the absolute token figures should not be read as a real model's context accounting.
- **A percentile of 100 samples is still a percentile of 100 samples.** `read` and `delete` earn a P95 by the rule this document sets itself, and 100 samples put roughly 5 above it. That is enough to say the tail exists and not enough to characterise it.
- **Two corpora, and only one of them is timed.** Every latency, cold-start and stage-split figure in this document is a property of the 7-document timed corpus. Every figure in the labelled-set sections is a property of the 24-document quality corpus, which no timed operation touches. The two must not be read across: a retrieval score there says nothing about the latency tables here.
- **Synthetic prose, real phrasings.** The labelled facts are generated from a small set of sentence templates over invented entity nouns, so the corpus can be shipped and re-derived byte for byte and so the labels cannot have leaked from anything a model saw in pretraining. Realism lives on the *query* side — terse fragments, verbose asks, misspellings — not in the text being retrieved. Real notes have different heading density and different vocabulary overlap, and this bench does not measure that.
- **The labelled set is English only.** By choice, not by oversight: the embedder is an English model, so a Cyrillic recall number would measure its vocabulary rather than chunk geometry. The multilingual half of the accepted review is reported as a probe in its own section and never enters a ranking. Nothing here says how a multilingual embedder would behave.
- **A held-out split mitigates tuning exposure; it does not undo it.** The three profiles existed before this labelled set did, so the held-out subset is genuinely unseen by profile selection — but the *corpus generator* was written by the same hand as the profiles, and a synthetic body cannot be blind in the way a real one would be.
- **The labelled scores carry reproduction noise, and the figure for it is a lower bound.** The quality corpus is large enough to engage the vector store's approximate index, so rebuilding it can reorder near-ties. Measured, not assumed — see *Are these scores reproducible?* — and the largest shift one rebuild caused in any held-out headline metric on any profile is 0.039682. That is one rebuild in one run: a separate run may move a labelled figure further. Only differences well clear of it should be read as differences. The timed corpus is too small for any of this to arise, which is why the equal-k quality block on that corpus reproduces exactly.
- **42 held-out answerable queries is still 42 queries.** One query moves the mean by 0.0238. Every separation claim in this document is tested against that figure, and a gap smaller than it is reported as noise rather than as a result.
- **No vendor comparison.** Every profile here is the same local stack. A hosted retrieval service is a different measurement and is not one of these rows.

The corpus, the labels and the result files are all in this repository, so any of the above can be checked rather than believed.
