# Topic-Modeling Methodology

This document assumes familiarity with embeddings, clustering, and BERTopic. For a plain-language explanation of what topic modeling is doing and why, see [`overview.md`](overview.md). For how the input transcripts these approaches consume are produced, see [`data-acquisition.md`](data-acquisition.md).

Approaches are classified primarily by **chunking strategy** — how the transcript gets grouped into units before embedding. Speaker-category information can additionally be attached to a chunking strategy as metadata (for filtering, coloring, or `topics_per_class` analysis) without that counting as a separate approach in its own right, since the chunk boundaries themselves haven't changed.

All approaches operate on transcripts already normalized to the `[MM:SS] SPEAKER: dialogue` format described in `data-acquisition.md`, with speaker labels mapped to coarse categories (e.g. `BEREAVED`, `PROFESSIONAL`, `EYEWITNESS`) prior to modeling.

> **Status:** two chunking approaches implemented so far. More will be added to this document as the research design expands.

| # | Name | Intention | How it works |
|---|------|-----------|---------------|
| **1** | Time-window | Baseline. Establish what BERTopic finds using a simple, speaker-agnostic chunking method — the reference point everything else is measured against. Speaker category can optionally be attached to each chunk afterward as metadata (the dominant coarse category within the window), for filtering/coloring results or running `topics_per_class` — this doesn't change the chunking itself, so it isn't a separate approach. | Fixed 60-second windows of transcript text, concatenated regardless of who's speaking. Embed each window, fit one global BERTopic model (UMAP + HDBSCAN + CountVectorizer). |
| **2** | Speaker-turn | Tests whether chunking along speaker-turn boundaries (instead of a clock) produces more coherent topics, at the cost of many more, shorter chunks. Here, speaker category is intrinsic to how chunks are formed, not just attached afterward. | One chunk per contiguous run of lines from the same fine-grained speaker label (e.g. `EYEWITNESS_02` speaking for 5 lines in a row = 1 chunk). No merging across turns, no merging across speaker changes. |
