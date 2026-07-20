# Benchmarks

Generated from the result files in `bench/results/` by `python bench.py --render-benchmarks`. **Do not edit by hand.** Every figure below is read out of one of those JSON files; a number in this document that is not derivable from a shipped result file is a defect.

Reproduce, from a fresh clone:

    ./install.sh && python bench.py --all --corpus bench/corpus

What produced the numbers below. **This table has no N and no mode of its own — it is where the N and the modes for every other table are declared.**

|  |  |
|---|---|
| source files | `bench/results/2026-07-20-chunk-small.json`, `bench/results/2026-07-20-baseline.json`, `bench/results/2026-07-20-chunk-large.json` |
| run date | 2026-07-20 |
| profiles | `chunk-small`, `baseline`, `chunk-large` |
| samples per operation | create **20**, read **100**, update **50**, delete **100**, cold start **5** |
| modes | `warm` (many operations in one process) and `cold-start` (a fresh process per sample), never mixed in one table |
| corpus | `bench/corpus/` — 7 files, 50699 characters, seed 20260720, 18 planted facts |
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
| create | 20 | 3665.45 | 1029.46 | 769.84 |
| read | 100 | 76.15 | 75.28 | 73.42 |
| update | 50 | 282.95 | 110.17 | 93.08 |
| delete | 100 | 9.52 | 8.46 | 9.61 |
| delete +lag | 100 | 16.81 | 15.60 | 15.52 |

Full distribution, milliseconds. `P50`/`P95` are blank where N < 100; there is no P99 column because no N here earns one:

| profile | operation | mode | n | median | MAD | min | max | P50 | P95 |
|---|---|---|---|---|---|---|---|---|---|
| chunk-small | create | warm | 20 | 3665.45 | 43.55 | 3601.49 | 4525.31 | - | - |
| chunk-small | read | warm | 100 | 76.15 | 0.99 | 73.91 | 139.51 | 76.15 | 84.24 |
| chunk-small | update | warm | 50 | 282.95 | 39.85 | 203.42 | 334.26 | - | - |
| chunk-small | delete | warm | 100 | 9.52 | 1.39 | 5.84 | 13.71 | 9.52 | 12.83 |
| chunk-small | delete +lag | warm | 100 | 16.81 | 0.39 | 15.53 | 54.94 | 16.81 | 22.28 |
| baseline | create | warm | 20 | 1029.46 | 4.91 | 1019.05 | 1149.56 | - | - |
| baseline | read | warm | 100 | 75.28 | 0.92 | 72.92 | 132.18 | 75.28 | 80.89 |
| baseline | update | warm | 50 | 110.17 | 8.39 | 94.19 | 157.30 | - | - |
| baseline | delete | warm | 100 | 8.46 | 1.42 | 4.99 | 21.00 | 8.46 | 13.19 |
| baseline | delete +lag | warm | 100 | 15.60 | 0.24 | 15.03 | 18.18 | 15.60 | 16.80 |
| chunk-large | create | warm | 20 | 769.84 | 6.84 | 757.06 | 822.54 | - | - |
| chunk-large | read | warm | 100 | 73.42 | 0.98 | 71.83 | 138.58 | 73.42 | 80.70 |
| chunk-large | update | warm | 50 | 93.08 | 6.77 | 79.82 | 148.07 | - | - |
| chunk-large | delete | warm | 100 | 9.61 | 1.66 | 5.19 | 19.91 | 9.61 | 13.21 |
| chunk-large | delete +lag | warm | 100 | 15.52 | 0.20 | 15.02 | 18.81 | 15.52 | 16.21 |

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
| chunk-small | process wall (total) | cold-start | 5 | 678.28 | 676.23 | 687.74 |
| chunk-small | init | cold-start | 5 | 489.44 | 484.02 | 498.08 |
| chunk-small | operation | cold-start | 5 | 82.32 | 81.17 | 89.68 |
| chunk-small | startup/teardown | cold-start | 5 | 107.30 | 99.98 | 110.93 |
| baseline | process wall (total) | cold-start | 5 | 669.06 | 662.79 | 685.81 |
| baseline | init | cold-start | 5 | 481.24 | 477.00 | 501.46 |
| baseline | operation | cold-start | 5 | 76.42 | 75.70 | 86.40 |
| baseline | startup/teardown | cold-start | 5 | 108.84 | 103.74 | 110.83 |
| chunk-large | process wall (total) | cold-start | 5 | 633.76 | 631.23 | 640.31 |
| chunk-large | init | cold-start | 5 | 454.52 | 449.24 | 456.35 |
| chunk-large | operation | cold-start | 5 | 75.76 | 73.83 | 79.83 |
| chunk-large | startup/teardown | cold-start | 5 | 104.77 | 102.15 | 110.76 |

`create` — a full index build, in a fresh process:

| profile | component | mode | n | median | min | max |
|---|---|---|---|---|---|---|
| chunk-small | process wall (total) | cold-start | 5 | 4094.72 | 4077.27 | 4121.64 |
| chunk-small | init | cold-start | 5 | 454.16 | 439.68 | 464.77 |
| chunk-small | operation | cold-start | 5 | 3526.74 | 3525.23 | 3549.41 |
| chunk-small | startup/teardown | cold-start | 5 | 111.63 | 107.18 | 119.15 |
| baseline | process wall (total) | cold-start | 5 | 1530.49 | 1523.60 | 1559.47 |
| baseline | init | cold-start | 5 | 450.22 | 444.64 | 455.02 |
| baseline | operation | cold-start | 5 | 962.61 | 956.64 | 991.52 |
| baseline | startup/teardown | cold-start | 5 | 117.34 | 113.72 | 123.31 |
| chunk-large | process wall (total) | cold-start | 5 | 1261.78 | 1249.59 | 1275.25 |
| chunk-large | init | cold-start | 5 | 447.11 | 438.71 | 451.75 |
| chunk-large | operation | cold-start | 5 | 703.03 | 700.21 | 710.90 |
| chunk-large | startup/teardown | cold-start | 5 | 116.10 | 108.06 | 117.25 |

On `chunk-small` the same read pass costs 76.15 ms warm and 678.28 ms from a fresh process — a factor of **9x**, essentially all of it paid before the first query is issued. v1 reported a `cold` column that was the first sample of a warm loop, in a process that had already imported everything and loaded the model; that column is gone.

**This is a cold *process*, not a cold *machine*.** The OS page cache still holds the ONNX model file after the first sample, and it is not purged between samples — purging it needs privileges this harness does not take. The first sample of the first profile pays a genuinely cold file read; the rest do not. Treat these figures as the cost of starting a new process on a machine that has run this before, which is the common case for a CLI, and not as a first-boot number.

## Where the read time goes

**mode `warm`, n=500 per profile** (100 passes over 5 fixed queries). Milliseconds per query.

| profile | stage | n | median | MAD | min | max | in the sum? |
|---|---|---|---|---|---|---|---|
| chunk-small | query embedding | 500 | 14.301 | 0.207 | 13.894 | 103.520 | yes |
| chunk-small | vector search | 500 | 1.411 | 0.234 | 1.059 | 46.457 | yes |
| chunk-small | result materialization | 500 | 0.019 | 0.019 | 0.000 | 8.821 | yes |
| chunk-small | **sum of the three** |  | 15.731 |  |  |  |  |
| chunk-small | **full in-process read** | 500 | 15.922 | 0.349 | 15.200 | 88.281 | is the total |
| chunk-small | full MCP round-trip | 20 | 83.351 | 3.144 | 78.806 | 96.237 | **no — envelope** |
| baseline | query embedding | 500 | 14.411 | 0.133 | 13.879 | 53.405 | yes |
| baseline | vector search | 500 | 0.396 | 0.036 | 0.316 | 5.187 | yes |
| baseline | result materialization | 500 | 0.094 | 0.031 | 0.000 | 0.567 | yes |
| baseline | **sum of the three** |  | 14.901 |  |  |  |  |
| baseline | **full in-process read** | 500 | 15.011 | 0.163 | 14.392 | 17.743 | is the total |
| baseline | full MCP round-trip | 20 | 80.778 | 1.162 | 78.328 | 90.698 | **no — envelope** |
| chunk-large | query embedding | 500 | 14.563 | 0.090 | 14.128 | 17.386 | yes |
| chunk-large | vector search | 500 | 0.348 | 0.022 | 0.294 | 0.728 | yes |
| chunk-large | result materialization | 500 | 0.081 | 0.022 | 0.000 | 0.465 | yes |
| chunk-large | **sum of the three** |  | 14.992 |  |  |  |  |
| chunk-large | **full in-process read** | 500 | 15.072 | 0.110 | 14.697 | 18.519 | is the total |
| chunk-large | full MCP round-trip | 20 | 84.901 | 4.155 | 78.702 | 97.673 | **no — envelope** |

How well the decomposition closes, and what had to be floored. Same samples as the table above — **mode `warm`, n=500 per profile** — restated so the residual is not read against a different N:

| profile | sum of stages | full read | residual | limit | materialization samples floored at 0 |
|---|---|---|---|---|---|
| chunk-small | 15.731 | 15.922 | 1.20% | 10% | 189 / 500 (37.8%) |
| baseline | 14.901 | 15.011 | 0.73% | 10% | 26 / 500 (5.2%) |
| chunk-large | 14.992 | 15.072 | 0.53% | 10% | 20 / 500 (4.0%) |

The residual is interpreter and call overhead that belongs to no stage. It is inside the 10% limit on every profile; a run where it is not fails and publishes nothing. The floored samples are the ones where materialization — the difference between the same search with and without its payload — came out negative through measurement noise; they are clamped to zero and counted here rather than silently.

**Do not read the materialization column as a measurement on `chunk-small`.** 37.8% of its samples floored, which means the stage is smaller than the run-to-run noise of the two searches it is the difference of. Its median is a lower bound on a quantity this clock cannot resolve, not an estimate of it. The number is left in the table because deleting it would make the three stages appear to sum more cleanly than they do; the two stages above it, and the total, are measurements.

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
| chunk-small | 1.000 | 6.88 | 32.50 | 1961.6 | 0.788 | 0.389 |
| baseline | 1.000 | 5.50 | 8.75 | 1942.5 | 0.361 | 0.492 |
| chunk-large | 0.938 | 4.25 | 6.50 | 1796.5 | 0.307 | 0.533 |

Budget **4000 tokens** — **untimed, so no mode applies; means over 8 labelled queries:**

| profile | recall@budget | unique docs | chunks admitted | tokens used | duplicate source rate | relevant token fraction |
|---|---|---|---|---|---|---|
| chunk-small | 1.000 | 7.00 | 66.12 | 3960.6 | 0.894 | 0.329 |
| baseline | 1.000 | 6.50 | 18.62 | 3880.5 | 0.647 | 0.390 |
| chunk-large | 1.000 | 6.00 | 15.62 | 3880.8 | 0.608 | 0.347 |

The unanswerable control was run under every budget on every profile and scored `recall@budget` 0 in all of them; a non-zero score there fails the run, exactly as it does at equal k.

## Findings

### 1. Read latency is query embedding, not index size

v1 claimed that on a small index the vector search is nearly free relative to MiniLM inference, and inferred it from the fact that read latency barely moved when the index size did. v2 measures it: on this corpus, **query embedding is 90–97% of the full in-process read** across the three profiles. The claim is now backed by the decomposition rather than by an absence of variation. Medians, **mode `warm`, n=500 per profile** — the same samples as [Where the read time goes](#where-the-read-time-goes), restated here:

| profile | chunks in index | query embedding (ms) | vector search (ms) | materialization (ms) | embedding share of the read |
|---|---|---|---|---|---|
| chunk-small | 256 | 14.301 | 1.411 | 0.019 | **89.8%** |
| baseline | 66 | 14.411 | 0.396 | 0.094 | **96.0%** |
| chunk-large | 48 | 14.563 | 0.348 | 0.081 | **96.6%** |

The vector search itself costs 0.348–1.411 ms while the number of chunks in the index spans **5.3x** (48 → 256). The embedding is a fixed cost per query that the chunking cannot move, and at this corpus size it dominates everything the index does. That is what makes read latency look flat: the median read pass spans only **1.04x** across the three profiles (73.42 ms on `chunk-large`, 76.15 ms on `chunk-small`), against a 6.7x span in the chunk budget and a 4.5x span in the mean chunk actually produced (239.2 → 1087.0 characters).

**The scope of that finding is this corpus.** 48 to 256 chunks is small enough that the ANN structure is doing almost nothing; the sentence above is about a small index, and nothing here measures where the crossover is.

### 2. Timing alone picks the wrong profile

On this corpus the timed operations do not separate the profiles on the axis a reader would care about. `chunk-large` is both the fastest to read and the cheapest to build, and if the tables stopped at latency it would look like the answer.

**Quality is what decides it.** At equal k, `baseline` leads on `recall@3` (1.000) and on `ndcg@10` (1.000), and it is the profile this toolkit ships as its default. The full ladder — note that this table deliberately puts three different N side by side: the latency columns are **mode `warm`**, read at **n=100** and create at **n=20**, while the quality columns are **untimed** means over **8 labelled queries**:

| profile | max chars | read median | create median | recall@3 | ndcg@10 |
|---|---|---|---|---|---|
| chunk-small | 300 | 76.15 | 3665.45 | 0.938 | 0.946 |
| baseline | 1200 | 75.28 | 1029.46 | 1.000 | 1.000 |
| chunk-large | 2000 | 73.42 | 769.84 | 0.708 | 0.900 |

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
- **No vendor comparison.** Every profile here is the same local stack. A hosted retrieval service is a different measurement and is not one of these rows.

The corpus, the labels and the result files are all in this repository, so any of the above can be checked rather than believed.
