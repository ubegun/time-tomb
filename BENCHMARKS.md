# Benchmarks

Generated from the result files in `bench/results/` by `python bench.py --render-benchmarks`. **Do not edit by hand.** Every figure below is read out of one of those JSON files; a number in this document that is not derivable from a shipped result file is a defect.

Reproduce, from a fresh clone:

    ./install.sh && python bench.py --all --corpus bench/corpus

|  |  |
|---|---|
| source files | `bench/results/2026-07-20-chunk-small.json`, `bench/results/2026-07-20-baseline.json`, `bench/results/2026-07-20-chunk-large.json` |
| run date | 2026-07-20 |
| profiles | `chunk-small`, `baseline`, `chunk-large` |
| runs per operation | 10 |
| corpus | `bench/corpus/` — 7 files, 50699 characters, seed 20260720, 18 planted facts |
| network | blocked; blocked outbound connects during the run: 0 |

## Environment

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

Four operations are timed, N=10 runs each, against a synthetic corpus generated deterministically from seed 20260720 and shipped in this repository as `bench/corpus/`. The corpus on disk is compared byte-for-byte with the generator's output before anything is measured, and the run fails loudly if they differ — otherwise the numbers would describe a corpus nobody can see.

| operation | what is timed |
|---|---|
| create | a full index build of the whole corpus |
| read | one pass over 5 fixed queries at k=5, fixed order, no sampling |
| update | re-index of a single changed note |
| delete | removal of one note |
| delete +lag | time until the removed note stops being returned by a probe query |

**Percentiles, not means.** A mean over N=10 runs hides the shape that matters here: the first run of any operation pays for a cold cache and the rest do not, so an average is a blend of two different things. P50 answers "what does this cost normally"; P95 answers "how bad is the tail"; the cold column is reported separately rather than being averaged into either.

**Quality is measured, not assumed.** Latency alone ranks a system that answers fast and wrong above one that answers slowly and right, so every run also scores document-level `recall@k` for k in 1, 3, 5, 10, plus MRR and `ndcg@10`, over 8 labelled queries planted into the corpus at generation time. Relevance is document-level: a retrieved chunk counts when its source file is in the query's relevant set. Chunk identity is the independent variable across these profiles, so chunk-keyed labels would make them incomparable by design. Scoring runs after all four timed operations and takes no timing samples.

**Two controls run every time, and are reported with their real numbers.** They exist because a check that can only pass measures nothing.

1. *An unanswerable query.* No document answers it, so it must score zero at every rung. It deliberately reuses vocabulary the corpus is full of, so the retriever answers it confidently with plausible chunks — the control has to survive a confident wrong answer, which is exactly the failure latency cannot see.
2. *Deranged labels.* Every query is scored against another query's documents, with the swap required to be document-disjoint from the truth. The aggregate must drop; if it does not, the labels are not independent of the retriever.

**Offline by design, and checked.** A socket guard is installed before anything heavy is imported; loopback and unix sockets stay allowed, every other outbound connect is refused and counted into the report. The runs below recorded 0, 0, 0 blocked attempts.

## Latency

P50, milliseconds:

| operation (P50 ms) | chunk-small | baseline | chunk-large |
|---|---|---|---|
| create | 3769.61 | 1083.94 | 815.52 |
| read | 80.12 | 79.14 | 78.83 |
| update | 292.26 | 112.13 | 97.58 |
| delete | 8.78 | 7.13 | 7.04 |
| delete +lag | 16.59 | 16.25 | 16.29 |

Full distribution, milliseconds:

| profile | operation | n | cold | P50 | P95 | P99 |
|---|---|---|---|---|---|---|
| chunk-small | create | 10 | 4248.62 | 3769.61 | 4036.77 | 4206.25 |
| chunk-small | read | 10 | 150.91 | 80.12 | 119.57 | 144.64 |
| chunk-small | update | 10 | 289.82 | 292.26 | 304.90 | 306.16 |
| chunk-small | delete | 10 | 15.30 | 8.78 | 14.09 | 15.06 |
| chunk-small | delete +lag | 10 | 17.28 | 16.59 | 17.12 | 17.25 |
| baseline | create | 10 | 1093.14 | 1083.94 | 1092.39 | 1092.99 |
| baseline | read | 10 | 147.78 | 79.14 | 117.29 | 141.68 |
| baseline | update | 10 | 166.63 | 112.13 | 146.40 | 162.59 |
| baseline | delete | 10 | 6.30 | 7.13 | 8.14 | 8.33 |
| baseline | delete +lag | 10 | 16.01 | 16.25 | 17.00 | 17.31 |
| chunk-large | create | 10 | 830.54 | 815.52 | 825.90 | 829.61 |
| chunk-large | read | 10 | 145.30 | 78.83 | 117.44 | 139.73 |
| chunk-large | update | 10 | 156.60 | 97.58 | 132.12 | 151.71 |
| chunk-large | delete | 10 | 6.95 | 7.04 | 7.77 | 7.86 |
| chunk-large | delete +lag | 10 | 16.10 | 16.29 | 16.99 | 17.01 |

Corpus as each profile cut it:

| profile | max chars | overlap | files | chunks | mean chunk | largest chunk |
|---|---|---|---|---|---|---|
| chunk-small | 300 | 40 | 7 | 256 | 239.2 | 310 |
| baseline | 1200 | 150 | 7 | 66 | 823.8 | 1211 |
| chunk-large | 2000 | 250 | 7 | 48 | 1087.0 | 1997 |

## Retrieval quality

| profile | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|
| chunk-small | 0.396 | 0.938 | 1.000 | 1.000 | 0.917 | 0.946 |
| baseline | 0.458 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chunk-large | 0.396 | 0.708 | 0.875 | 1.000 | 0.893 | 0.900 |

`recall@1` cannot reach 1.0: with 2 to 3 relevant documents per query (mean 2.25), one retrieved chunk can cover at most one of them, which caps the mean at **0.458**.

### Controls

| profile | unanswerable rr | unanswerable ndcg@10 | docs it still returned | deranged label overlap | MRR true → deranged | recall@3 true → deranged |
|---|---|---|---|---|---|---|
| chunk-small | 0.000 | 0.000 | 5 | 0 | 0.917 → 0.351 | 0.938 → 0.312 |
| baseline | 0.000 | 0.000 | 6 | 0 | 1.000 → 0.266 | 1.000 → 0.250 |
| chunk-large | 0.000 | 0.000 | 6 | 0 | 0.893 → 0.308 | 0.708 → 0.354 |

The unanswerable query scored zero on every profile while the retriever confidently returned several distinct documents for it. Deranged labels are document-disjoint from the truth — overlap 0 — and every profile dropped.

## Finding

**Read latency is flat, and timing alone picks the wrong profile.** Across the three profiles the P50 of a read pass spans **1.02x** (78.83 ms on `chunk-large`, 80.12 ms on `chunk-small`), while the chunk budget spans **6.7x** (300 → 2000 characters), the mean chunk actually produced spans **4.5x** (239.2 → 1087.0 characters) and the number of chunks in the index spans **5.3x** (48 → 256). A 6.7x change in how the corpus is cut moved read latency by 2%.

That is the unflattering part of the method, stated first: on this corpus the timed operations do not separate the profiles on the axis that a reader would care about. `chunk-large` is both the fastest to read and the cheapest to build, and if the tables stopped at latency it would look like the answer.

**Quality is what decides it.** `baseline` leads on `recall@3` (1.000) and on `ndcg@10` (1.000), and it is the profile this toolkit ships as its default. The full ladder:

| profile | max chars | read P50 | create P50 | recall@3 | ndcg@10 |
|---|---|---|---|---|---|
| chunk-small | 300 | 80.12 | 3769.61 | 0.938 | 0.946 |
| baseline | 1200 | 79.14 | 1083.94 | 1.000 | 1.000 |
| chunk-large | 2000 | 78.83 | 815.52 | 0.708 | 0.900 |

Read the table as a decision, not as a ranking: the coarsest profile buys its speed with retrieval quality, the finest one costs more to build *and* scores lower, and the difference is invisible to every timed operation above.

## Limitations

These numbers are worth exactly what their scope allows, and the scope is small.

- **One machine, one session.** Every figure comes from the single environment in the table above. Nothing here is a cross-machine claim, and the absolute milliseconds will not transfer.
- **A small synthetic corpus.** 7 documents, 50699 characters, 8 labelled queries with 2 to 3 relevant documents each. That resolves the *direction* of the quality difference; it does not pin its magnitude, and a handful of queries carry the coarse profile's deficit.
- **Synthetic, not natural, text.** The corpus is generated so that it can be shipped and re-derived byte-for-byte. Real notes have different heading density and different vocabulary overlap, both of which the chunker is sensitive to.
- **Local only, single client, no concurrency.** One process, one collection, no parallel readers or writers, no server hop. Nothing here says anything about behaviour under load.
- **One embedder.** chroma-onnx / all-MiniLM-L6-v2 throughout. The comparison is between chunk geometries at a fixed embedder, not between embedders.
- **Cold and warm are both present.** The `cold` column is the first run of each operation and is reported separately, but the process itself is warm by the time later profiles run; ordering effects at the millisecond scale are not controlled for.
- **No vendor comparison.** Every profile here is the same local stack. A hosted retrieval service is a different measurement and is not one of these rows.

The corpus, the labels and the result files are all in this repository, so any of the above can be checked rather than believed.
