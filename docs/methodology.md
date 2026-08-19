# Topic-Modeling Methodology

This document assumes familiarity with embeddings, clustering, and BERTopic. For a plain-language explanation of what topic modeling is doing and why, see [`overview.md`](overview.md). For how the input transcripts these approaches consume are produced, see [`data-acquisition.md`](data-acquisition.md).

Approaches are classified primarily by **chunking strategy** — how the transcript gets grouped into units before embedding. Speaker-category information can additionally be attached to a chunking strategy as metadata (for filtering, coloring, or `topics_per_class` analysis) without that counting as a separate approach in its own right, since the chunk boundaries themselves haven't changed.

All approaches operate on transcripts already normalized to the `[MM:SS] SPEAKER: dialogue` format described in `data-acquisition.md`, with speaker labels mapped to coarse categories (e.g. `BEREAVED`, `PROFESSIONAL`, `EYEWITNESS`) prior to modeling.

> **Status:** two chunking approaches implemented so far. More will be added to this document as the research design expands.

| # | Name | Intention | How it works |
|---|------|-----------|---------------|
| **1** | Time-window | Baseline. Establish what BERTopic finds using a simple, speaker-agnostic chunking method — the reference point everything else is measured against. Speaker category can optionally be attached to each chunk afterward as metadata (the dominant coarse category within the window), for filtering/coloring results or running `topics_per_class` — this doesn't change the chunking itself, so it isn't a separate approach. | Fixed 60-second windows of transcript text, concatenated regardless of who's speaking. Embed each window, fit one global BERTopic model (UMAP + HDBSCAN + CountVectorizer). |
| **2** | Speaker-turn (turn-strict) | Tests whether chunking along speaker-turn boundaries (instead of a clock) produces more coherent topics, at the cost of many more, shorter chunks. Here, speaker category is intrinsic to how chunks are formed, not just attached afterward. | One chunk per contiguous run of lines from the same fine-grained speaker label (e.g. `EYEWITNESS_02` speaking for 5 lines in a row = 1 chunk). No merging across turns, no merging across speaker changes. |

## Parsing notes

Speaker labels are extracted from raw transcript lines with a regex-based parser shared across both approaches. A known artifact — a second, nested speaker label sometimes leaking into the captured dialogue text (e.g. `SPEAKER_01: EYEWITNESS_02: I think what happened...`) — is stripped in a second pass (`strip_leaked_speaker_label`) after the primary speaker match, since the two labels can't be distinguished in a single regex pass (both sit before the first colon the primary match can see).

## Clustering: parameter sweep, not a single fixed configuration

Earlier versions of each approach fit one BERTopic model per notebook at a single, hand-picked `(min_cluster_size, min_samples)` HDBSCAN setting. Both approaches now instead run a **grid sweep** across:

- `min_cluster_size`: 5, 10, 15, 20, 25
- `min_samples`: 1, 3, 5

producing 15 distinct topic-model configurations per approach, rather than one. This exists because the "right" granularity for HDBSCAN isn't knowable in advance, and it turns out to matter a lot: the two chunking strategies do **not** behave comparably at the same settings. At `min_cluster_size=5, min_samples=1`, for example, time-window (2,785 chunks) finds 12 topics, while turn-strict (6,270 chunks) finds 176 — driven by turn-strict's larger corpus size and its chunks being shorter and more semantically concentrated (no cross-speaker blending to smooth chunks toward each other), which lets HDBSCAN separate more, smaller clusters. This is expected behavior given the two chunking designs, not a bug in either notebook — see the sweep cell's inline notes for the fuller explanation.

**Efficiency note:** UMAP dimensionality reduction is fit once, on the full embedding set, and reused across all 15 configs — only HDBSCAN's clustering step (which is cheap) varies per config. This also isolates differences between configs to the clustering step itself, rather than confounding them with a fresh (and only nominally identical) UMAP refit per config.

Each config additionally runs BERTopic's automatic topic reduction (`nr_topics="auto"`) as a post-processing step, which merges topics whose c-TF-IDF keyword profiles are highly similar. This is a conservative merge — it does not force topics down to a specific target count, only combines genuinely near-duplicate topics — so raw topic counts (e.g. 176) can still be high after this step if the underlying clusters are meaningfully distinct.

Sweep results are exported as two JSON files per approach (`{notebook_name}_shared_data.json`, `{notebook_name}_sweep_configs.json`), which power an interactive slider UI in the HTML output — letting a reader explore how topic granularity changes across the parameter grid without re-running anything.

## Topic labeling

Each topic's raw BERTopic name (a `word1_word2_word3`-style label from top c-TF-IDF keywords, e.g. `0_gun_people_nra_guns`) is relabeled with a short, human-readable name generated by a local LLM (`gemma-4-E4B-it-text-only`), run once per topic per sweep configuration. The prompt includes the topic's weighted keywords plus a small set of representative example chunks — the chunks closest to the topic's embedding centroid, i.e. its most prototypical members — so the label reflects actual transcript language, not just keyword co-occurrence. Both the LLM-generated label and BERTopic's original raw name are retained (the raw name is shown in the HTML output as a secondary note, for transparency about how the label was derived).

A small set of representative example chunks (by default, the 3 chunks closest to each topic's centroid) is also retained per topic per config, for display in the HTML output, so a reader can check a topic's label against real transcript passages rather than trusting keywords alone. To avoid this scaling with `topics × configs × corpus size`, example chunk text is deduplicated globally: a lookup table stores each selected chunk's text once, regardless of how many topics/configs reference it, and topic records store only chunk indices into that table.

## Outlier handling

Chunks HDBSCAN doesn't confidently assign to any topic are marked as outliers (`topic_id = -1`) rather than forced into a nearest cluster. An optional outlier-reduction step (reassigning outliers to their nearest topic by embedding similarity, via BERTopic's `reduce_outliers`) exists in both notebooks but is currently disabled by default — outlier rate is reported as a data-quality statistic rather than something that gets minimized by construction. Turn-strict shows a meaningfully higher outlier rate than time-window at comparable settings, largely because it retains many short, low-content turns (single-word backchannels like "Right." or "Mm-hmm.") that survive the word-count floor but carry little topical signal.

## Algorithmic speaker-class validation (turn-strict only)

Turn-strict's HTML output includes an additional, opt-in analysis not present in time-window: individual speakers are clustered by their own language alone (via centroid-pooled embeddings + HDBSCAN), with zero knowledge of the coarse categories assigned during transcript labeling. This is only valid when a chunk is guaranteed to belong to exactly one speaker, which time-window's chunking doesn't guarantee (a single time window can blend multiple speakers together, so a "speaker centroid" built from it wouldn't represent any one person). Agreement between the algorithmic clusters and the assigned categories is measured with adjusted Rand index (ARI) — in the current turn-strict run, ARI is close to zero against both assigned category and film identity, indicating the algorithm isn't recovering either signal from language alone. This is treated as a genuine (if inconclusive) research finding rather than something to force into agreement.