"""
export_html.py — shared HTML export for the BERTopic speaker-category project.

Builds the polished, interactive documentary-connections page (film
similarity map, top pairs, theme breadth, category breakdown, per-film
theme/category pies) from whatever results_df / topic_model a given
approach produced.

This module is shared across approach notebooks (1&2, 3, 4, 5, ...). Each
notebook does its own chunking + BERTopic fit, producing its own results_df
and topic_model, then calls export_documentary_html(...) with those objects.
Nothing in this file is approach-specific — it only assumes results_df has
the columns: documentary, category, topic_id, text_chunk (and that
topic_model / new_labels can resolve topic_id -> a human label). text_chunk
is needed by the Film Similarity Map section, which re-embeds every chunk
to compute real per-film centroid vectors -- every approach notebook's
results_df already carries this column. If a future approach's results_df
has a different shape (e.g. #6's per-category independent models, which
have no single shared topic space), this module will likely need a variant
rather than a drop-in call — see the note at the bottom of this file.

Usage (last cell of any approach notebook):

    from export_html import export_documentary_html

    export_documentary_html(
        results_df=results_df,
        topic_model=topic_model,
        new_labels=new_labels,
        embedding_model=embedding_model,
        d3_js=d3_js,
        notebook_name='approach_4_turn_merge_coarse',  # -> visualizations/approach_4_turn_merge_coarse.html
        strategy_label='Turn-Merge-Coarse, Speaker-Structure',  # names this
                                                                  # approach in
                                                                  # the hero
    )

All approach notebooks write into one shared `visualizations/` folder (the
`output_dir` default) with a filename derived from `notebook_name`, so every
approach's page lives side by side without collisions or everyone
overwriting the same file.
"""

import os
import json
import difflib
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap
import hdbscan


# Collapses the ~50 raw speaker-derived labels (e.g. BEREAVED_ADVOCATE_REFORM_03,
# JACKSON KATZ, FRAT BOY) into a fixed set of coarse buckets. First keyword match
# wins, in this priority order — so a compound label like BEREAVED_ADVOCATE_REFORM
# collapses into BEREAVED rather than getting its own bucket.
CATEGORY_PRIORITY = [
    ('NARRATOR', 'NARRATOR'),
    ('BEREAVED', 'BEREAVED'),
    ('ADVOCATE_PROGUN', 'ADVOCATE_PROGUN'),
    ('ADVOCATE_REFORM', 'ADVOCATE_REFORM'),
    ('COMMUNITY_VOICE', 'COMMUNITY_VOICE'),
    ('PROFESSIONAL', 'PROFESSIONAL'),
    ('NEWS_CLIP', 'NEWS_CLIP'),
    ('FAMILY_FRIEND', 'FAMILY_FRIEND'),
]


def collapse_category(raw_category: str) -> str:
    upper = raw_category.upper()
    for keyword, bucket in CATEGORY_PRIORITY:
        if keyword in upper:
            return bucket
    return 'OTHER'


def fuzzy_match_title(doc_name: str, lookup: dict, char_threshold: float = 0.85,
                       word_threshold: float = 0.6):
    """Matches a results_df['documentary'] value (a transcript filename stem,
    e.g. 'Bowling_for_Columbine_Transcript') against a dict keyed by the
    master documentary list's display titles (e.g. 'Bowling for Columbine').

    Requires BOTH of two containment scores to pass, not just one:

    1. Character containment: how much of the SHORTER normalized string's
       characters are found in the longer one. Needed because the master
       list's titles often carry a descriptive subtitle the actual filenames
       drop entirely (e.g. "The Brutal Truth: A Violence Documentary" vs. a
       filename that's just "The_Brutal_Truth") -- plain difflib ratio()
       scores that pair ~0.58 since it penalizes the length difference, so a
       naive ratio threshold can't tell a real match like this apart from a
       real non-match.

    2. Word containment: how many of the SHORTER string's actual WORDS (not
       just characters, trailing 's' stripped so singular/plural still
       count) appear in the longer one. This exists because character
       containment alone has its own failure mode: 'Run Hide Fight' vs.
       'Gun Fight' scores 0.889 on character containment alone (they share
       " fight" plus "un" from "r-UN-" / "g-UN-"), which would silently
       mismatch two unrelated films. Word containment correctly drops that
       same pair to 0.5, well under threshold, while still passing genuine
       matches like the subtitle case above and 'Quiet Rooms' vs.
       'Quiet Room'. Verified against real title/filename collisions from
       this project, not just synthetic examples.

    Returns the matched value from `lookup`, or None if either check fails
    -- callers should treat None as "no data," not as a signal to guess,
    since a low-confidence guess is exactly what this pipeline is designed
    to keep out.
    """
    def normalize(s: str) -> str:
        s = s.replace('_Transcript', '').replace('_transcript', '')
        s = s.replace('_', ' ').replace('-', ' ')
        s = ''.join(ch for ch in s.lower() if ch.isalnum() or ch == ' ')
        return ' '.join(s.split())

    def char_containment(a: str, b: str) -> float:
        sm = difflib.SequenceMatcher(None, a, b)
        matched = sum(block.size for block in sm.get_matching_blocks())
        return matched / max(1, min(len(a), len(b)))

    def word_containment(a: str, b: str) -> float:
        def tokens(s: str) -> set:
            return {w.rstrip('s') for w in s.split()}
        ta, tb = tokens(a), tokens(b)
        shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if not shorter:
            return 0.0
        return len(shorter & longer) / len(shorter)

    target = normalize(doc_name)
    best_key, best_combined, best_char, best_word = None, -1.0, 0.0, 0.0
    for key in lookup:
        norm_key = normalize(key)
        c_score = char_containment(target, norm_key)
        w_score = word_containment(target, norm_key)
        # Jointly optimize both scores -- selecting by char alone first (as an
        # earlier version of this function did) let a coincidental char-only
        # match mask a real self-match sitting elsewhere in the lookup: e.g.
        # 'Tower' scored a perfect 1.0 char containment against the unrelated
        # 'CHI-TOWN GUNS...' (several short fragments -- "tow", "n g", etc --
        # happened to sum to 100% of "tower"'s 5 characters) and, being
        # evaluated first in iteration order, that tie was never displaced by
        # the genuine 'Tower' self-match also scoring 1.0. Taking the MINIMUM
        # of the two scores as the ranking criterion fixes this: a key only
        # wins if it's simultaneously good on both axes, so a coincidental
        # character-only overlap can't outrank a real match.
        combined = min(c_score, w_score)
        if combined > best_combined:
            best_key, best_combined, best_char, best_word = key, combined, c_score, w_score

    if best_char >= char_threshold and best_word >= word_threshold:
        return lookup[best_key]
    return None


def build_film_similarity_payload(
    all_chunks: list,
    all_doc_names: list,
    all_categories: list,
    embedding_model,
    embeddings,
    random_state: int = 42,
) -> dict:
    """Builds the data for the Film Similarity Map section, replacing the old
    force-directed network graph (which was built on binary topic-presence
    vectors, not actual embeddings, and whose spring layout visually implies
    clustering regardless of whether the underlying similarity matrix has
    real block structure).

    This directly answers two pieces of feedback: (1) "how did you aggregate
    from embeddings to a film-level similarity?" -- explicit centroid
    pooling, stated both in code and in the method_note shown on the page;
    and (2) "networks look beautiful but don't reveal real group structure"
    -- replaced with an actual UMAP projection plus a real cluster-validity
    test (HDBSCAN + silhouette sweep) that can report "no structure found"
    instead of a layout that always looks clustered by construction.

    Args:
        all_chunks: every chunk's raw text -- the SAME list BERTopic was fit
            on for this approach notebook (same chunking, unchanged).
        all_doc_names: per-chunk documentary name, same length as all_chunks.
        all_categories: per-chunk dominant/coarse speaker category (already
            collapsed via collapse_category(), for consistency with the rest
            of this module's categorical breakdowns), same length as
            all_chunks.
        embedding_model: the SentenceTransformer already used to fit
            BERTopic for this approach. Kept as a required argument (used
            below for topic-label embeddings elsewhere in this module) but
            deliberately NOT used to re-encode chunks here -- see
            `embeddings` below for why.
        embeddings: the EXACT embeddings array already used to fit BERTopic
            for this approach notebook (same order as all_chunks). This is
            intentionally a required argument with no "recompute if missing"
            fallback. An earlier version of this function re-encoded chunks
            from scratch when embeddings weren't passed in, on the
            assumption that re-encoding the same text with the same model
            "deterministically reproduces the same vectors." That assumption
            is false in practice: two encode() calls with different
            batch_size/device settings differ by ~1e-6 per element (ordinary
            floating-point noise), and for this dataset that noise is enough
            to change which side of a near-tie UMAP's optimizer lands on --
            measured directly at up to 0.71 on a [-1, 1]-normalized axis,
            i.e. a materially different layout, not jitter. UMAP itself is
            deterministic given fixed input (confirmed empirically), so the
            fix is to guarantee there is only ever one embeddings array per
            run, computed once upstream in the notebook and threaded through
            everywhere downstream, rather than letting this function
            silently produce its own slightly-different copy.
        random_state: shared seed for UMAP / KMeans / HDBSCAN.

    Returns:
        dict with: films (per-film x/y/cluster/category/chunk-count
        records), cluster_quality (silhouette sweep + HDBSCAN result),
        method_note (the aggregation explanation rendered directly on the
        page), n_films.
    """
    df = pd.DataFrame({'text': all_chunks, 'doc': all_doc_names, 'category': all_categories})

    # ── Centroid pooling: average each film's chunk embeddings ─────────────
    # This is the aggregation step. Computed on raw embeddings in their
    # original dimensionality (e.g. 768-d for all-mpnet-base-v2) -- not on a
    # topic-presence vector and not after any reduction. `embeddings` must
    # already be the array used to fit BERTopic for this approach -- no
    # recompute fallback (see docstring above for why that was removed).
    if embeddings is None:
        raise ValueError(
            "build_film_similarity_payload() requires the exact `embeddings` "
            "array used to fit BERTopic for this approach -- pass your "
            "notebook's `embeddings` variable explicitly. Re-encoding chunks "
            "here instead of reusing that array has been shown to shift the "
            "resulting UMAP layout by up to 0.71 on a normalized axis, even "
            "with the same model, same text, and the same random_state."
        )
    if len(embeddings) != len(all_chunks):
        raise ValueError(
            f"embeddings has {len(embeddings)} rows but all_chunks has "
            f"{len(all_chunks)} entries -- they must be the same length and "
            f"in the same order (embeddings[i] must correspond to "
            f"all_chunks[i]), or centroid pooling below will silently "
            f"average together the wrong chunks for a given film."
        )

    films = sorted(df['doc'].unique())
    centroids = np.zeros((len(films), embeddings.shape[1]))
    chunk_counts, dominant_category, dominant_category_pct = {}, {}, {}
    for i, film in enumerate(films):
        mask = (df['doc'] == film).values
        centroids[i] = embeddings[mask].mean(axis=0)
        chunk_counts[film] = int(mask.sum())
        cat_counts = df.loc[mask, 'category'].value_counts()
        dominant_category[film] = str(cat_counts.idxmax())
        # Purity of the dominant label, not just which label won the plurality --
        # a film that's 85% one category is a much stronger read than one that's
        # 20% its "dominant" category with everything else spread thin. Used to
        # drive node size on the map so a big node means "trust this label."
        dominant_category_pct[film] = round(100 * float(cat_counts.max()) / mask.sum(), 1)

    n_films = len(films)

    # ── Cluster validity on the ORIGINAL high-dim centroids ─────────────────
    # Deliberately not on the 2D UMAP projection below: UMAP optimizes for
    # visual neighbor structure, not metric fidelity, so cluster counts or
    # silhouette scores computed on its output would test the visualization
    # rather than the data.
    silhouette_by_k = []
    max_k = min(8, n_films - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(centroids)
        sil = silhouette_score(centroids, km.labels_, metric='cosine')
        silhouette_by_k.append({'k': k, 'silhouette': round(float(sil), 3)})
    best = max(silhouette_by_k, key=lambda r: r['silhouette']) if silhouette_by_k else None

    # HDBSCAN with a permissive min_cluster_size given how few films there
    # typically are -- the point is to let the data say "no real grouping"
    # via the noise label (-1) when that's genuinely the case, which KMeans
    # cannot do (it always partitions every point into some cluster).
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    hdbscan_labels = clusterer.fit_predict(centroids)
    n_noise = int((hdbscan_labels == -1).sum())
    n_clusters_found = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)

    # ── UMAP to 2D for the scatter plot only ────────────────────────────────
    # n_neighbors must be < n_samples; the UMAP default of 15 silently
    # misbehaves once n_films drops below ~16, so it's scaled down
    # explicitly rather than left at the library default.
    n_neighbors = max(2, min(15, n_films - 1))
    reducer = umap.UMAP(
        n_components=2, metric='cosine', random_state=random_state,
        n_neighbors=n_neighbors, min_dist=0.15,
    )
    coords = reducer.fit_transform(centroids)
    coords = coords - coords.mean(axis=0)
    scale = np.abs(coords).max() or 1.0
    coords = coords / scale

    film_records = [
        {
            'doc': film,
            'x': round(float(coords[i, 0]), 4),
            'y': round(float(coords[i, 1]), 4),
            'chunks': chunk_counts[film],
            'dominant_category': dominant_category[film],
            'dominant_category_pct': dominant_category_pct[film],
            'hdbscan_cluster': int(hdbscan_labels[i]),
        }
        for i, film in enumerate(films)
    ]

    method_note = (
        f"Each film is reduced to a single vector by averaging the embeddings of all its "
        f"transcript chunks (centroid pooling) in the original {embeddings.shape[1]}-dimensional "
        f"embedding space. UMAP then projects those {n_films} film centroids to 2D for display "
        f"only; cluster membership and the silhouette scores below are computed on the original "
        f"high-dimensional centroids, not the 2D layout, since UMAP does not preserve distances."
    )

    return {
        'films': film_records,
        'cluster_quality': {
            'silhouette_by_k': silhouette_by_k,
            'best_k': best['k'] if best else None,
            'best_silhouette': best['silhouette'] if best else None,
            'hdbscan_n_clusters': n_clusters_found,
            'hdbscan_n_noise': n_noise,
            'hdbscan_noise_pct': round(100 * n_noise / n_films, 1) if n_films else 0.0,
        },
        'method_note': method_note,
        'n_films': n_films,
    }


def build_chunk_scatter_payload(
    all_chunks: list,
    all_doc_names: list,
    all_categories: list,
    embeddings,
    random_state: int = 42,
    max_points: int = 6000,
) -> dict:
    """
    Point-per-chunk UMAP scatter -- the finest granularity the professor's
    note calls out ("each point can be a time-chunk, speaker turn, etc."),
    complementary to the film-level map in build_film_similarity_payload.

    The film map can only ever show structure that survives averaging every
    chunk in a film down to one centroid -- it's incapable of showing
    whether individual moments of speech actually separate by speaker
    category, only whether the *average* does. This function answers that
    more fundamental question directly: does raw per-chunk language, with
    no aggregation at all, separate by category in embedding space.

    A fresh, independent UMAP fit on the raw per-chunk embeddings -- NOT a
    zoomed-in view of the film-level map, which was fit on film centroids
    and has no notion of individual chunks.

    Args:
        all_chunks: chunk texts, same order as embeddings.
        all_doc_names: film name per chunk, same order/length as all_chunks.
        all_categories: coarse speaker category per chunk, same order.
        embeddings: the exact array used to fit BERTopic for this approach
            (same requirement and same reasoning as build_film_similarity_
            payload -- see that function's docstring for why there is no
            recompute fallback here either).
        random_state: UMAP seed.
        max_points: chunk-level counts vary a lot across the 10 chunking
            approaches (turn-strict and semantic-boundary chunking in
            particular can produce far more, far shorter chunks than
            fixed-window approaches) -- above this many points, render
            performance and legibility both degrade, so a random subsample
            is drawn instead. The full chunk count is still reported
            alongside so it's clear from the page itself when this
            happened, rather than silently rendering a subset.

    Returns:
        dict with: points (per-chunk x/y/doc/category/snippet records),
        n_total_chunks, n_shown, sampled (bool).
    """
    if embeddings is None:
        raise ValueError(
            "build_chunk_scatter_payload() requires the exact `embeddings` "
            "array used to fit BERTopic for this approach -- see "
            "build_film_similarity_payload()'s docstring for why there's no "
            "recompute fallback."
        )
    if len(embeddings) != len(all_chunks):
        raise ValueError(
            f"embeddings has {len(embeddings)} rows but all_chunks has "
            f"{len(all_chunks)} entries -- they must be the same length and "
            f"in the same order."
        )

    n = len(all_chunks)
    if n > max_points:
        # Stratified by film rather than a global random draw. A plain random
        # sample is proportional to each film's length, so a handful of long
        # films would dominate the point cloud and effectively define what
        # e.g. "BEREAVED" looks like on the map just by chunk-count volume --
        # not because their speech is representative of bereaved speakers in
        # general. Giving every film an equal quota instead means the
        # category-separation question gets tested fairly across films,
        # rather than being skewed toward whichever documentaries are
        # longest. Short films under quota simply contribute all their
        # chunks (so total shown can land a bit under max_points -- that's
        # expected, not a bug).
        rng = np.random.RandomState(random_state)
        docs_arr = np.asarray(all_doc_names)
        unique_docs = sorted(set(all_doc_names))
        per_film_quota = max(1, max_points // len(unique_docs))
        idx_parts = []
        for doc in unique_docs:
            doc_idx = np.where(docs_arr == doc)[0]
            if len(doc_idx) > per_film_quota:
                doc_idx = rng.choice(doc_idx, size=per_film_quota, replace=False)
            idx_parts.append(doc_idx)
        idx = np.sort(np.concatenate(idx_parts))
    else:
        idx = np.arange(n)

    sub_embeddings = embeddings[idx]

    # ── Fresh UMAP fit on raw per-chunk embeddings ──────────────────────────
    # Independent of the film-level UMAP fit -- different input (per-chunk,
    # not per-film-centroid), so a different n_neighbors scaling and its own
    # random_state usage, but same principle: 2D projection for display only.
    n_points = len(idx)
    n_neighbors = max(2, min(15, n_points - 1))
    reducer = umap.UMAP(
        n_components=2, metric='cosine', random_state=random_state,
        n_neighbors=n_neighbors, min_dist=0.1,
    )
    coords = reducer.fit_transform(sub_embeddings)
    coords = coords - coords.mean(axis=0)
    scale = np.abs(coords).max() or 1.0
    coords = coords / scale

    points = []
    for row_i, i in enumerate(idx):
        snippet = str(all_chunks[i]).strip().replace('\n', ' ')
        if len(snippet) > 100:
            snippet = snippet[:97] + '...'
        points.append({
            'x': round(float(coords[row_i, 0]), 4),
            'y': round(float(coords[row_i, 1]), 4),
            'doc': all_doc_names[i],
            'category': all_categories[i],
            'snippet': snippet,
        })

    return {
        'points': points,
        'n_total_chunks': n,
        'n_shown': n_points,
        'sampled': n > max_points,
    }


def build_character_class_payload(
    all_chunks: list,
    all_doc_names: list,
    all_speakers: list,
    all_categories: list,
    embeddings,
    random_state: int = 42,
    min_chunks_per_speaker: int = 3,
) -> dict:
    """
    Algorithmic character-class map: the professor's second, separate ask
    ("what different character classes do you identify algorithmically")
    which is NOT the same thing as the a priori speaker categories used
    everywhere else on this page. Those categories were assigned, not
    discovered. This clusters individual speakers by their own language and
    reports what an unsupervised method finds on its own -- then measures
    how well that agrees with the assigned categories, rather than assuming
    agreement.

    ONLY valid for chunking strategies that guarantee a chunk belongs to
    exactly one speaker (e.g. turn-strict). Time-window chunking can blend
    multiple speakers into a single chunk, which would make a "speaker
    centroid" built from it a mix of multiple people's language, not a
    profile of any one person -- silently wrong rather than loudly wrong,
    which is exactly the failure mode to avoid. There is deliberately no
    auto-detection of this from the data; the caller (export_documentary_
    html's chunks_are_single_speaker flag) must assert it explicitly.

    Grouping key is (doc, speaker), not speaker alone: fine-grained speaker
    labels like 'BEREAVED_ADVOCATE_REFORM_05' are assigned per-transcript by
    the upstream diarization step and are NOT globally unique -- the same
    label in two different films almost certainly refers to two different
    real people who happened to get the same auto-generated index.

    Args:
        all_chunks, all_doc_names, all_categories: same shape/meaning as in
            build_film_similarity_payload (all_categories should already be
            the collapsed coarse category).
        all_speakers: fine-grained per-chunk speaker label, same order.
        embeddings: the exact array used to fit BERTopic for this approach
            (same requirement as the other two builders in this module).
        min_chunks_per_speaker: speakers below this many chunks are dropped
            before clustering -- a centroid built from 1-2 chunks is mostly
            noise, and including them would degrade the UMAP/HDBSCAN fit for
            everyone else without adding real signal. How many got dropped
            this way is reported, not silently discarded.

    Returns:
        dict with: characters (per-speaker x/y/doc/speaker/category/
        hdbscan_cluster/chunk_count records), cluster_quality (silhouette
        sweep + HDBSCAN result, same shape as the film map's), agreement
        (adjusted Rand index between the a priori categories and the
        algorithmically discovered HDBSCAN clusters -- the actual answer to
        "do the assigned categories match what the data shows"), n_excluded
        (speakers dropped for too few chunks), method_note.
    """
    if embeddings is None:
        raise ValueError(
            "build_character_class_payload() requires the exact `embeddings` "
            "array used to fit BERTopic for this approach -- see "
            "build_film_similarity_payload()'s docstring for why there's no "
            "recompute fallback."
        )
    if len(embeddings) != len(all_chunks):
        raise ValueError(
            f"embeddings has {len(embeddings)} rows but all_chunks has "
            f"{len(all_chunks)} entries -- they must be the same length and "
            f"in the same order."
        )

    df = pd.DataFrame({
        'doc': all_doc_names, 'speaker': all_speakers, 'category': all_categories,
    })
    df['character_id'] = df['doc'] + ' :: ' + df['speaker']

    counts = df['character_id'].value_counts()
    keep_ids = counts[counts >= min_chunks_per_speaker].index
    n_excluded = int((counts < min_chunks_per_speaker).sum())

    characters = sorted(keep_ids)
    n_chars = len(characters)
    if n_chars < 3:
        raise ValueError(
            f"Only {n_chars} speakers have >= {min_chunks_per_speaker} chunks -- "
            f"not enough to cluster meaningfully. Lower min_chunks_per_speaker "
            f"or check that `all_speakers` is actually the fine-grained "
            f"per-chunk speaker column, not the coarse category."
        )

    centroids = np.zeros((n_chars, embeddings.shape[1]))
    chunk_counts, doc_of, speaker_of, category_of = {}, {}, {}, {}
    for i, cid in enumerate(characters):
        mask = (df['character_id'] == cid).values
        centroids[i] = embeddings[mask].mean(axis=0)
        chunk_counts[cid] = int(mask.sum())
        row0 = df.loc[mask].iloc[0]
        doc_of[cid] = row0['doc']
        speaker_of[cid] = row0['speaker']
        category_of[cid] = row0['category']

    # ── Cluster validity on the ORIGINAL high-dim centroids ─────────────────
    # Same reasoning as build_film_similarity_payload: HDBSCAN's noise label
    # (-1) lets the data say "this person doesn't cleanly belong to any
    # group" rather than forcing every speaker into some class.
    silhouette_by_k = []
    max_k = min(8, n_chars - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(centroids)
        sil = silhouette_score(centroids, km.labels_, metric='cosine')
        silhouette_by_k.append({'k': k, 'silhouette': round(float(sil), 3)})
    best = max(silhouette_by_k, key=lambda r: r['silhouette']) if silhouette_by_k else None

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    hdbscan_labels = clusterer.fit_predict(centroids)
    n_noise = int((hdbscan_labels == -1).sum())
    n_clusters_found = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)

    # ── Agreement between a priori category and algorithmic cluster ────────
    # The actual answer to "what character classes do you identify
    # algorithmically, and do they match what you assigned." 1.0 = perfect
    # agreement, 0.0 = no better than chance, negative = worse than chance.
    # Computed including noise (-1) as its own label, since "the algorithm
    # couldn't place this person anywhere" is itself part of the comparison.
    a_priori = [category_of[cid] for cid in characters]
    ari = adjusted_rand_score(a_priori, hdbscan_labels.tolist())

    # ── Agreement between algorithmic cluster and FILM identity ─────────────
    # A second, different comparison, not a variant of the one above: tests
    # whether HDBSCAN's clusters are actually tracking speaker category at
    # all, or are really just rediscovering "which documentary is this" --
    # a single film's specific vocabulary (named victims, places, events)
    # can dominate an embedding far more than an abstract stylistic category
    # like BEREAVED vs PROFESSIONAL ever would. If this comes back high while
    # ari (above) stays near zero, that's a quantified confirmation that the
    # clusters are film fingerprints, not character classes -- not something
    # to infer by eye from which colors happen to share a film in the plot.
    film_labels = [doc_of[cid] for cid in characters]
    ari_vs_film = adjusted_rand_score(film_labels, hdbscan_labels.tolist())

    # ── UMAP to 2D for the scatter plot only ────────────────────────────────
    n_neighbors = max(2, min(15, n_chars - 1))
    reducer = umap.UMAP(
        n_components=2, metric='cosine', random_state=random_state,
        n_neighbors=n_neighbors, min_dist=0.15,
    )
    coords = reducer.fit_transform(centroids)
    coords = coords - coords.mean(axis=0)
    scale = np.abs(coords).max() or 1.0
    coords = coords / scale

    character_records = [
        {
            'id': cid,
            'doc': doc_of[cid],
            'speaker': speaker_of[cid],
            'category': category_of[cid],
            'x': round(float(coords[i, 0]), 4),
            'y': round(float(coords[i, 1]), 4),
            'chunks': chunk_counts[cid],
            'hdbscan_cluster': int(hdbscan_labels[i]),
        }
        for i, cid in enumerate(characters)
    ]

    method_note = (
        f"Each point is one individual speaker (identified by film + fine-grained "
        f"diarization label), centroid-pooled from their own {embeddings.shape[1]}-"
        f"dimensional chunk embeddings. {n_excluded} speaker(s) with fewer than "
        f"{min_chunks_per_speaker} chunks were excluded as too sparse to profile "
        f"reliably. Cluster membership, silhouette scores, and the agreement scores "
        f"below are computed on the original high-dimensional centroids, not the 2D "
        f"layout. Shape encodes the category you assigned; color encodes the cluster "
        f"HDBSCAN found on its own. Two agreement numbers are reported: how well "
        f"clusters match your assigned category, and how well they match plain film "
        f"identity -- a high film-agreement alongside a low category-agreement means "
        f"the algorithm is mostly rediscovering which documentary a speaker is from "
        f"(driven by that film's specific vocabulary), not finding a stylistic "
        f"character class independent of film."
    )

    # ── Per-cluster purity, film and category ───────────────────────────────
    # The two ARI numbers above are GLOBAL averages -- dominated by whatever
    # the largest mass of points does, so a few small, genuinely pure
    # clusters can get statistically swamped by one big undifferentiated
    # blob and still show up as "no agreement" overall. This breaks that
    # same comparison down cluster-by-cluster instead, which is the only way
    # to see whether a handful of small clusters are real film- or
    # category-fingerprints even while the aggregate signal looks like noise.
    cluster_breakdown = []
    all_cluster_ids = sorted(set(hdbscan_labels.tolist()))
    for cid in all_cluster_ids:
        member_idx = [i for i, cluster in enumerate(hdbscan_labels) if cluster == cid]
        size = len(member_idx)
        member_films = [film_labels[i] for i in member_idx]
        member_cats = [a_priori[i] for i in member_idx]
        film_counts = Counter(member_films)
        cat_counts = Counter(member_cats)
        top_film, top_film_n = film_counts.most_common(1)[0]
        top_cat, top_cat_n = cat_counts.most_common(1)[0]
        cluster_breakdown.append({
            'cluster': int(cid),
            'size': size,
            'dominant_film': top_film,
            'dominant_film_pct': round(100 * top_film_n / size, 1),
            'n_distinct_films': len(film_counts),
            'dominant_category': top_cat,
            'dominant_category_pct': round(100 * top_cat_n / size, 1),
            'n_distinct_categories': len(cat_counts),
        })
    # Noise (-1) first if present (it's not a real cluster, worth seeing
    # separately), then real clusters largest-to-smallest.
    cluster_breakdown.sort(key=lambda r: (r['cluster'] != -1, -r['size']))

    return {
        'characters': character_records,
        'cluster_quality': {
            'silhouette_by_k': silhouette_by_k,
            'best_k': best['k'] if best else None,
            'best_silhouette': best['silhouette'] if best else None,
            'hdbscan_n_clusters': n_clusters_found,
            'hdbscan_n_noise': n_noise,
            'hdbscan_noise_pct': round(100 * n_noise / n_chars, 1) if n_chars else 0.0,
        },
        'agreement': {
            'adjusted_rand_index': round(float(ari), 3),
            'adjusted_rand_index_vs_film': round(float(ari_vs_film), 3),
        },
        'cluster_breakdown': cluster_breakdown,
        'method_note': method_note,
        'n_characters': n_chars,
        'n_excluded': n_excluded,
    }


def shared_topics(df_clean, doc1, doc2, top_n=5):
    d1 = set(df_clean[df_clean.documentary == doc1]['topic_label'].unique())
    d2 = set(df_clean[df_clean.documentary == doc2]['topic_label'].unique())
    counts = []
    for t in d1 & d2:
        c1 = int(df_clean[(df_clean.documentary == doc1) & (df_clean.topic_label == t)].shape[0])
        c2 = int(df_clean[(df_clean.documentary == doc2) & (df_clean.topic_label == t)].shape[0])
        counts.append({"label": t, "doc1_chunks": c1, "doc2_chunks": c2, "total": c1 + c2})
    return sorted(counts, key=lambda x: -x["total"])[:top_n]


def build_data_payload(results_df, topic_model, new_labels, embedding_model, embeddings=None,
                        chunks_are_single_speaker=False,
                        production_type_map=None, streaming_category_map=None):
    """Runs the full analysis (steps 1-8 of the original cell) and returns
    the `data` dict that the HTML template consumes. Pulled out as its own
    function so it can be unit-tested or inspected without touching disk.

    chunks_are_single_speaker: set True only for chunking strategies that
        guarantee each chunk is exactly one speaker's turn (e.g. turn-strict,
        turn-merge-coarse, hybrid-capped) -- required to enable the
        algorithmic character-class map (see build_character_class_payload's
        docstring for why time-window chunking can't validly support it).
        When True, results_df must also carry a 'speaker' column with the
        fine-grained per-chunk speaker label. Defaults to False so this is
        an explicit opt-in per notebook, not an auto-detected guess.
    production_type_map, streaming_category_map: optional dicts keyed by the
        master documentary list's display titles (e.g. 'Bowling for
        Columbine'), mapping to 'Independent' / 'Studio/Network-backed' and
        'Major Platform' / 'Free Streaming' / 'YouTube' / 'Other'
        respectively. Matched against results_df['documentary'] via fuzzy
        title matching (see fuzzy_match_title) since filenames and titles
        aren't guaranteed identical. IMPORTANT: production_type_map should
        already have any low-confidence classifications filtered out before
        being passed in -- a film missing from the dict is rendered as
        genuinely unclassified (no border) rather than guessed, and that
        filtering is meant to happen upstream, not here.
    """

    # ── 1. Topic labels + speaker categories ───────────────────────────────
    topic_info_lookup = {
        row['Topic']: new_labels.get(row['Topic'], row['Name'])
        for _, row in topic_model.get_topic_info().iterrows()
        if row['Topic'] != -1
    }

    # ── 1b. Semantic theme colors ───────────────────────────────────────────
    # Embed each topic's label with the same model used for clustering, reduce
    # to one dimension with PCA, then map that dimension onto a 0-320 degree
    # hue range (320 instead of 360 avoids the wheel wrapping back to the same
    # red at both ends). Themes whose labels are conceptually similar end up
    # with similar embeddings, hence nearby PC1 values, hence nearby — not
    # identical, but visibly closer — hues. This is a 1D projection of
    # high-dimensional meaning, so it captures the dominant axis of variation
    # across your topics, not every notion of "similar"; two themes can be
    # close in meaning along an axis PC1 doesn't emphasize and still land a
    # fair distance apart in hue.
    topic_labels_ordered = [topic_info_lookup[t] for t in sorted(topic_info_lookup)]
    topic_label_embeddings = embedding_model.encode(topic_labels_ordered)

    pca_hue = PCA(n_components=1, random_state=42)
    pc1 = pca_hue.fit_transform(topic_label_embeddings).ravel()
    lo, hi = pc1.min(), pc1.max()
    hues = (pc1 - lo) / (hi - lo or 1) * 320
    theme_hue = {label: round(float(h), 1) for label, h in zip(topic_labels_ordered, hues)}

    df_clean = results_df[results_df.topic_id != -1].copy()
    df_clean['topic_label'] = df_clean['topic_id'].map(topic_info_lookup)
    df_clean['coarse_category'] = df_clean['category'].apply(collapse_category)

    presence = (
        df_clean.groupby(['documentary', 'topic_label'])
        .size().gt(0).unstack(fill_value=0).astype(int)
    )
    presence = presence.loc[
        sorted(presence.index),
        presence.sum().sort_values(ascending=False).index
    ]

    sim = cosine_similarity(presence.values)
    np.fill_diagonal(sim, 0)
    sim_df = pd.DataFrame(sim, index=presence.index, columns=presence.index)

    # ── 2. All pairs ─────────────────────────────────────────────────────────
    pairs_raw = (
        sim_df.where(np.triu(np.ones(sim_df.shape), k=1).astype(bool))
        .stack().sort_values(ascending=False)
    )
    top_pairs = [
        {"doc1": d1, "doc2": d2, "score": round(float(s), 4)}
        for (d1, d2), s in pairs_raw.items()
    ]

    # ── 3. Per-doc top topics ───────────────────────────────────────────────
    # Percentage is of this film's own classified (non-outlier) chunk count,
    # so short and long films are comparable — a raw count would always
    # favor longer films. Every theme at or above OTHER_THRESHOLD_PCT gets its
    # own slice (no cap on how many — a film with 11 themes above the
    # threshold shows 11 slices), and everything below that gets pooled into
    # one 'Other' entry that still carries its own theme count and combined
    # percentage (so hovering it reads e.g. "Other — 7 films, 4.2%" rather
    # than being an opaque leftover with no detail behind it).
    OTHER_THRESHOLD_PCT = 2.0
    doc_topics = {}
    for doc in presence.index:
        doc_total = int((df_clean.documentary == doc).sum())
        counts = (
            df_clean[df_clean.documentary == doc]
            .groupby('topic_label').size().sort_values(ascending=False)
        )
        rows = []
        other_count = 0
        other_n_themes = 0
        for k, v in counts.items():
            pct = round(100 * v / doc_total, 1) if doc_total else 0.0
            if pct >= OTHER_THRESHOLD_PCT:
                rows.append({"label": k, "count": int(v), "pct": pct})
            else:
                other_count += int(v)
                other_n_themes += 1
        if other_count > 0:
            rows.append({
                "label": "Other",
                "count": int(other_count),
                "pct": round(100 * other_count / doc_total, 1) if doc_total else 0.0,
                "n_themes": other_n_themes,
            })
        doc_topics[doc] = rows

    # ── 3b. Per-doc category breakdown ──────────────────────────────────────
    # Same shape as doc_topics, but grouped by coarse speaker category instead
    # of theme. collapse_category() is exhaustive (every chunk lands in one of
    # the named buckets or OTHER), so unlike doc_topics this never needs an
    # "Other" slice to reach 100% — there's no analogous threshold-pooling step
    # here since there are only 9 categories total.
    doc_categories = {}
    for doc in presence.index:
        doc_total = int((df_clean.documentary == doc).sum())
        counts = (
            df_clean[df_clean.documentary == doc]
            .groupby('coarse_category').size().sort_values(ascending=False)
        )
        doc_categories[doc] = [
            {"label": k, "count": int(v), "pct": round(100 * v / doc_total, 1) if doc_total else 0.0}
            for k, v in counts.items()
        ]

    # ── 4. Shared topics for top 30 pairs ───────────────────────────────────
    top_pairs_detail = []
    for p in top_pairs[:30]:
        top_pairs_detail.append({**p, "shared": shared_topics(df_clean, p["doc1"], p["doc2"])})

    # ── 5. Topic summary ─────────────────────────────────────────────────────
    topic_doc_counts = presence.sum().reset_index()
    topic_doc_counts.columns = ['label', 'doc_count']
    topic_summary = [
        {"label": row['label'], "doc_count": int(row['doc_count'])}
        for _, row in topic_doc_counts.iterrows()
    ]

    # ── 6. Film Similarity Map (replaces the old network graph) ────────────
    # Uses the chunks' actual text and the SAME embedding_model that fit
    # BERTopic for this approach, so similarity is based on real centroid
    # embeddings rather than the binary topic-presence vectors that drove
    # `sim`/`top_pairs` above. results_df already carries text_chunk per
    # chunk (every approach notebook builds it that way), so no notebook
    # changes are needed to supply this.
    film_similarity = build_film_similarity_payload(
        all_chunks=results_df['text_chunk'].tolist(),
        all_doc_names=results_df['documentary'].tolist(),
        all_categories=results_df['category'].apply(collapse_category).tolist(),
        embedding_model=embedding_model,
        embeddings=embeddings,
    )

    # ── 6b. Attach dominant theme onto each film map record ─────────────────
    # doc_topics (built in step 3 above) is BERTopic's actual per-film theme
    # breakdown, sorted descending by count and already excluding outlier
    # (-1) chunks -- this is the real topic-model result the map has not
    # been using at all until now (it previously only ever saw speaker
    # category). dominant_theme_pct is that top entry's own pct field, i.e.
    # what fraction of the film's classified chunks landed in its single
    # most common theme -- the "how dominant is the dominant label" purity
    # score, reused here as the map's size channel.
    for f in film_similarity['films']:
        top_themes = doc_topics.get(f['doc'], [])
        if top_themes:
            f['dominant_theme'] = top_themes[0]['label']
            f['dominant_theme_pct'] = top_themes[0]['pct']
        else:
            # Film had no non-outlier chunks at all -- shouldn't normally
            # happen, but fail soft rather than KeyError downstream in JS.
            f['dominant_theme'] = None
            f['dominant_theme_pct'] = 0.0

    # ── 6b2. Attach production type / streaming platform (optional) ─────────
    # Fuzzy-matched against the master documentary list since transcript
    # filenames and the list's display titles aren't guaranteed identical.
    # A film with no match (or whose production classification was filtered
    # out upstream for low confidence) gets None here -- rendered as
    # genuinely unclassified on the map, not guessed.
    if production_type_map or streaming_category_map:
        for f in film_similarity['films']:
            f['production_type'] = (
                fuzzy_match_title(f['doc'], production_type_map) if production_type_map else None
            )
            f['streaming_category'] = (
                fuzzy_match_title(f['doc'], streaming_category_map) if streaming_category_map else None
            )
    else:
        for f in film_similarity['films']:
            f['production_type'] = None
            f['streaming_category'] = None

    # ── 6c. Chunk-level scatter (finest granularity) ─────────────────────────
    # Complementary to film_similarity above: one point per chunk, no
    # aggregation at all, so it can show whether speaker categories actually
    # separate in raw embedding space rather than only after averaging.
    chunk_scatter = build_chunk_scatter_payload(
        all_chunks=results_df['text_chunk'].tolist(),
        all_doc_names=results_df['documentary'].tolist(),
        all_categories=results_df['category'].apply(collapse_category).tolist(),
        embeddings=embeddings,
    )

    # ── 6d. Algorithmic character-class map (opt-in, see docstring above) ───
    character_class = None
    if chunks_are_single_speaker:
        if 'speaker' not in results_df.columns:
            raise ValueError(
                "chunks_are_single_speaker=True requires results_df to have a "
                "'speaker' column (the fine-grained per-chunk speaker label) "
                "-- it wasn't found."
            )
        character_class = build_character_class_payload(
            all_chunks=results_df['text_chunk'].tolist(),
            all_doc_names=results_df['documentary'].tolist(),
            all_speakers=results_df['speaker'].tolist(),
            all_categories=results_df['category'].apply(collapse_category).tolist(),
            embeddings=embeddings,
        )

    # ── 7. Speaker category breakdown ───────────────────────────────────────
    # Full coarse_category x topic_label cross-tab, row-normalized to % of
    # that category's own classified chunks (the only direction the UI uses).
    cat_topic_counts = pd.crosstab(df_clean['coarse_category'], df_clean['topic_label'])
    cat_topic_pct = cat_topic_counts.div(cat_topic_counts.sum(axis=1), axis=0).mul(100).round(1)

    category_breakdown = {
        "categories": sorted(cat_topic_counts.index.tolist()),
        "by_category": {
            cat: sorted(
                [
                    {"label": topic, "count": int(cat_topic_counts.loc[cat, topic]),
                     "pct": float(cat_topic_pct.loc[cat, topic])}
                    for topic in cat_topic_counts.columns
                    if cat_topic_counts.loc[cat, topic] > 0
                ],
                key=lambda x: -x["count"]
            )
            for cat in cat_topic_counts.index
        },
    }

    # ── 6b. Network Edges for Hybrid Map ────────────────────────────────────
    network_edges = [
        {"source": p["doc1"], "target": p["doc2"], "score": p["score"]}
        for p in top_pairs if p["score"] > 0
    ]

    # ── 8. Full data payload ────────────────────────────────────────────────
    return {
        "docs": sorted(presence.index.tolist()),
        "doc_topics": doc_topics,
        "doc_categories": doc_categories,
        "topic_summary": topic_summary,
        "top_pairs": top_pairs_detail,
        "film_similarity": film_similarity,
        "chunk_scatter": chunk_scatter,
        "character_class": character_class,
        "network_edges": network_edges,         # <-- Add this line
        "category_breakdown": category_breakdown,
        "theme_hue": theme_hue,
        "n_topics": int(len(topic_info_lookup)),
        "n_chunks": int(len(results_df)),
        "n_docs": int(len(presence.index)),
    }


def render_html(data, d3_js, strategy_label='Time-Window, Speaker-Metadata'):
    """Pure string-templating step: takes the data payload and returns the
    full HTML document as a string. No file I/O here — see
    export_documentary_html() for the part that writes to disk.

    strategy_label appears in the hero eyebrow (e.g. "BERTopic · Cosine
    Similarity · <strategy_label>"), naming which chunking/modeling approach
    produced this particular page. Each approach notebook should pass its
    own short, human-readable name here — e.g. 'Turn-Strict, Speaker-
    Structure' for #3, 'Turn-Merge-Coarse' for #4, 'Hybrid-Capped' for #5 —
    so the page never silently describes the wrong methodology, which is
    what happened when the old static methodology section was reused as-is
    across approaches that chunk completely differently."""

    # ── Character-class map: conditional, only when the notebook opted in
    # via chunks_are_single_speaker=True (see build_character_class_payload's
    # docstring). Built as plain (non-f) strings so they can be embedded into
    # the main f-string below via simple substitution, without their own
    # literal JS/CSS braces needing to be doubled the way the rest of this
    # function's template text does.
    character_nav_link = ''
    character_section_html = ''
    character_map_js = ''
    if data.get('character_class'):
        character_nav_link = '  <a href="#charactermap">Character Map</a>\n'
        character_section_html = """
    <!-- 01F ALGORITHMIC CHARACTER-CLASS MAP -->
    <section id="charactermap">
      <div class="section-header">
        <span class="section-num">01F</span>
        <h2>Character Classes (Algorithmic)</h2>
      </div>
      <p class="section-desc">
        Every point here is one individual speaker — not a category you assigned, but centroid-pooled
        from their own chunk embeddings and then clustered by HDBSCAN with no knowledge of your
        speaker-category labels at all. <strong>Shape</strong> encodes the category you assigned;
        <strong>color</strong> encodes the cluster the algorithm found on its own; <strong>size</strong>
        reflects how many chunks that speaker has, a genuine proxy for speaking time at the individual
        level. If a shape is consistently one color, the algorithm agrees with your labeling; if a shape
        scatters across colors, it's finding different structure than you assumed. The number that
        actually settles that question is the adjusted Rand index in the panel alongside — not
        eyeballing color/shape overlap, which is easy to over- or under-read by eye.
      </p>
      <div class="filmmap-layout" style="display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap;">
        <div class="filmmap-wrap" style="flex: 1; min-width: 320px;">
          <svg id="charmap-svg"></svg>
          <div class="filmmap-controls" style="padding: 16px; background: var(--bg2); display: flex; flex-direction: column; gap: 10px; align-items: flex-start;">
            <div class="filmmap-legend" id="charmap-legend-shape" style="margin-left: 0; display: flex; gap: 16px; flex-wrap: wrap;"></div>
            <div class="filmmap-legend" id="charmap-legend-color" style="margin-left: 0; display: flex; gap: 16px; flex-wrap: wrap;"></div>
          </div>
        </div>
        <div class="quality-panel" id="charmap-quality-panel"></div>
      </div>
      <div id="charmap-cluster-table" style="margin-top: 20px;"></div>
    </section>
"""
        character_map_js = """
// ── CHARACTER-CLASS MAP (algorithmic, D3 scatter) ───────────────────────────
try {
(function() {
  const cc = DATA.character_class;
  if (!cc || !cc.characters || !cc.characters.length) return;
  const chars = cc.characters;

  const svg = d3.select('#charmap-svg');
  const container = document.getElementById('charmap-svg').parentElement;
  const W = container.offsetWidth || 900, H = 600;
  svg.attr('viewBox', `0 0 ${W} ${H}`);

  const PAD = 50;
  const xScale = d3.scaleLinear().domain(d3.extent(chars, d => d.x)).range([PAD, W - PAD]).nice();
  const yScale = d3.scaleLinear().domain(d3.extent(chars, d => d.y)).range([H - PAD, PAD]).nice();
  const rScale = d3.scaleSqrt().domain(d3.extent(chars, d => d.chunks)).range([5, 16]);

  const gGrid = svg.append('g');
  gGrid.selectAll('line.h').data(yScale.ticks(5)).enter().append('line')
    .attr('x1', PAD).attr('x2', W - PAD).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
    .attr('stroke', '#1f1f1f').attr('stroke-width', 1);
  gGrid.selectAll('line.v').data(xScale.ticks(5)).enter().append('line')
    .attr('y1', PAD).attr('y2', H - PAD).attr('x1', d => xScale(d)).attr('x2', d => xScale(d))
    .attr('stroke', '#1f1f1f').attr('stroke-width', 1);

  const SYMBOL_TYPES = [
    d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond,
    d3.symbolCross, d3.symbolStar, d3.symbolWye,
  ];
  const cats = Array.from(new Set(chars.map(d => d.category))).sort();
  function shapeTypeFor(cat) { return SYMBOL_TYPES[cats.indexOf(cat) % SYMBOL_TYPES.length]; }

  const HUE_PALETTE = [12, 200, 45, 280, 150, 320, 90, 260];
  const clusterIds = Array.from(new Set(chars.map(d => d.hdbscan_cluster))).filter(c => c !== -1).sort((a,b) => a-b);
  function clusterColor(cid) {
    if (cid === -1) return 'hsl(0, 0%, 40%)';
    return `hsl(${HUE_PALETTE[clusterIds.indexOf(cid) % HUE_PALETTE.length]}, 45%, 52%)`;
  }

  function tooltipHtml(d) {
    const clusterLabel = d.hdbscan_cluster === -1 ? 'No cluster (noise)' : `Cluster ${d.hdbscan_cluster}`;
    return `<b>${cleanName(d.doc)}</b><br>Speaker: ${d.speaker}<br>` +
      `Assigned category: ${d.category}<br>Algorithmic: ${clusterLabel}<br>${d.chunks} chunks`;
  }

  const gPoints = svg.append('g');
  gPoints.selectAll('path.char-node').data(chars, d => d.id).join('path')
    .attr('class', 'char-node')
    .attr('transform', d => `translate(${xScale(d.x)},${yScale(d.y)})`)
    .attr('d', d => {
      const r = rScale(d.chunks);
      return d3.symbol().type(shapeTypeFor(d.category)).size(Math.PI * r * r)();
    })
    .attr('fill', d => d.hdbscan_cluster === -1 ? 'none' : clusterColor(d.hdbscan_cluster))
    .attr('fill-opacity', 0.85)
    .attr('stroke', d => d.hdbscan_cluster === -1 ? 'hsl(0,0%,55%)' : '#e6e2d8')
    .attr('stroke-width', d => d.hdbscan_cluster === -1 ? 1.2 : 0.75)
    .attr('stroke-dasharray', d => d.hdbscan_cluster === -1 ? '2,2' : null)
    .style('cursor', 'pointer')
    .on('mouseenter', function(event, d) {
      d3.select(this).attr('stroke-width', 2);
      showTip(event, tooltipHtml(d));
    })
    .on('mouseleave', function() {
      d3.select(this).attr('stroke-width', d => d.hdbscan_cluster === -1 ? 1.2 : 0.75);
      hideTip();
    });

  svg.call(d3.zoom().scaleExtent([0.5, 4]).on('zoom', e => { gPoints.attr('transform', e.transform); }));

  const shapeLegend = document.getElementById('charmap-legend-shape');
  shapeLegend.innerHTML = '<div class="legend-item" style="color:var(--muted)"><b>Shape</b> = assigned category</div>';
  cats.forEach(cat => {
    const d = d3.symbol().type(shapeTypeFor(cat)).size(70)();
    shapeLegend.innerHTML += `<div class="legend-item">` +
      `<svg width="14" height="14" style="overflow:visible;vertical-align:middle;margin-right:5px">` +
      `<path d="${d}" transform="translate(7,7)" fill="none" stroke="#c9c3b8" stroke-width="1.4"></path>` +
      `</svg>${cat}</div>`;
  });

  const colorLegend = document.getElementById('charmap-legend-color');
  colorLegend.innerHTML = '<div class="legend-item" style="color:var(--muted)"><b>Color</b> = algorithmic cluster</div>';
  clusterIds.forEach(cid => {
    colorLegend.innerHTML += `<div class="legend-item"><div class="legend-dot" style="background:${clusterColor(cid)}"></div>Cluster ${cid}</div>`;
  });
  if (chars.some(d => d.hdbscan_cluster === -1)) {
    colorLegend.innerHTML += `<div class="legend-item"><div class="legend-dot" style="background:transparent;border:1.5px dashed hsl(0,0%,55%)"></div>No cluster (noise)</div>`;
  }

  const q = cc.cluster_quality;
  const ari = cc.agreement.adjusted_rand_index;
  const ariFilm = cc.agreement.adjusted_rand_index_vs_film;
  let verdict, verdictOk;
  // The film-identity comparison takes priority when it's the dominant
  // signal: a cluster set that mostly rediscovers "which documentary" isn't
  // finding character classes at all, regardless of how the category number
  // alone might otherwise read.
  if (ariFilm >= 0.25 && ariFilm > ari + 0.1) {
    verdict = `Clusters track film identity (${ariFilm}) far more than assigned category (${ari}) -- this looks like the algorithm mostly rediscovering which documentary a speaker is from (that film's specific vocabulary), not a stylistic character class independent of film.`;
    verdictOk = false;
  } else if (ari >= 0.5) {
    verdict = 'Strong agreement — the algorithm independently recovers structure close to your assigned categories.';
    verdictOk = true;
  } else if (ari >= 0.25) {
    verdict = "Moderate agreement — some real alignment, but the algorithm is also finding differences your category scheme doesn't capture.";
    verdictOk = true;
  } else if (ari >= 0) {
    verdict = "Weak to no agreement — the algorithm mostly isn't recovering your categories from language alone.";
    verdictOk = false;
  } else {
    verdict = 'Worse than chance — worth treating as a prompt to revisit the categories or the clustering, not proof either one is simply wrong.';
    verdictOk = false;
  }

  const panel = document.getElementById('charmap-quality-panel');
  panel.innerHTML = `
    <div class="quality-title">Cluster Validity</div>
    <div class="quality-stat"><div class="quality-stat-n">${cc.n_characters}</div><div class="quality-stat-label">Speakers profiled (${cc.n_excluded} excluded, too few chunks)</div></div>
    <div class="quality-stat"><div class="quality-stat-n">${q.hdbscan_n_clusters}</div><div class="quality-stat-label">Algorithmic clusters found</div></div>
    <div class="quality-stat"><div class="quality-stat-n">${q.hdbscan_noise_pct}%</div><div class="quality-stat-label">Speakers unassigned (noise)</div></div>
    <div class="quality-stat"><div class="quality-stat-n">${q.best_silhouette ?? '—'}</div><div class="quality-stat-label">Best silhouette (k=${q.best_k ?? '—'})</div></div>
    <div class="quality-verdict ${verdictOk ? 'ok' : ''}">
      vs. assigned category: <b>${ari}</b><br>
      vs. film identity: <b>${ariFilm}</b><br><br>${verdict}
    </div>`;

  // ── Per-cluster purity table ─────────────────────────────────────────────
  // The global ARIs above are averages across everyone -- a few small,
  // genuinely pure clusters can be completely real and still get
  // statistically swamped by one big undifferentiated blob, showing up as
  // "no agreement" overall. This is the direct, cluster-by-cluster check:
  // does THIS specific cluster look like a film fingerprint, a category
  // fingerprint, or neither.
  const tableEl = document.getElementById('charmap-cluster-table');
  if (tableEl && cc.cluster_breakdown && cc.cluster_breakdown.length) {
    const rows = cc.cluster_breakdown.map(r => {
      const label = r.cluster === -1 ? 'No cluster (noise)' : `Cluster ${r.cluster}`;
      const filmFlag = r.dominant_film_pct >= 70 ? ' ⚑' : '';
      const catFlag = r.dominant_category_pct >= 70 ? ' ⚑' : '';
      return `<tr>
        <td>${label}</td>
        <td>${r.size}</td>
        <td>${cleanName(r.dominant_film)} (${r.dominant_film_pct}%)${filmFlag} · ${r.n_distinct_films} film(s)</td>
        <td>${r.dominant_category} (${r.dominant_category_pct}%)${catFlag} · ${r.n_distinct_categories} categor${r.n_distinct_categories === 1 ? 'y' : 'ies'}</td>
      </tr>`;
    }).join('');
    tableEl.innerHTML = `
      <div class="quality-title" style="margin-bottom: 10px;">Per-Cluster Breakdown</div>
      <div style="font-size: 11px; color: var(--muted); margin-bottom: 10px;">
        ⚑ marks a cluster that's ≥70% one film or one category — a real fingerprint, not noise,
        even where the aggregate agreement scores above look weak.
      </div>
      <table style="width:100%; border-collapse: collapse; font-family: var(--mono); font-size: 11px;">
        <thead>
          <tr style="text-align:left; color: var(--muted); border-bottom: 1px solid var(--border);">
            <th style="padding:6px 10px;">Cluster</th>
            <th style="padding:6px 10px;">Size</th>
            <th style="padding:6px 10px;">Dominant film</th>
            <th style="padding:6px 10px;">Dominant category</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
    tableEl.querySelectorAll('td, th').forEach(cell => {
      cell.style.borderBottom = '1px solid var(--border)';
    });
  }
})();
} catch (err) {
  console.error('Character-class map failed to render:', err);
}

"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Documentary Connections — Gun Violence Films</title>
__D3_INJECT__
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');
:root {{
  --bg:     #0c0c0c; --bg2: #141414; --bg3: #1c1c1c;
  --border: #282828; --red: #b83232; --red2: #d94040;
  --gold:   #c8a04a; --text: #e6e2d8; --muted: #6e6860;
  --mono:   'IBM Plex Mono', monospace;
  --serif:  'Playfair Display', Georgia, serif;
  --sans:   'Inter', system-ui, sans-serif;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.6; -webkit-font-smoothing: antialiased; }}
a {{ color: inherit; text-decoration: none; }}

nav {{
  position: sticky; top: 0; z-index: 100;
  background: var(--bg); border-bottom: 1px solid var(--border);
  padding: 0 56px; display: flex; align-items: center; gap: 32px; height: 48px;
}}
.nav-brand {{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; color: var(--red); text-transform: uppercase; margin-right: 16px; }}
nav a {{ font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); padding: 4px 0; border-bottom: 1px solid transparent; transition: color 0.15s, border-color 0.15s; }}
nav a:hover {{ color: var(--text); border-color: var(--red); }}

.hero {{
  padding: 72px 56px 56px; border-bottom: 1px solid var(--border);
  display: grid; grid-template-columns: 1fr auto; gap: 40px; align-items: end;
}}
.hero-eyebrow {{ font-family: var(--mono); font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--red); margin-bottom: 16px; }}
.hero h1 {{ font-family: var(--serif); font-size: clamp(30px, 4vw, 52px); line-height: 1.12; font-weight: 700; max-width: 640px; }}
.hero h1 em {{ font-style: italic; color: var(--gold); }}
.hero-sub {{ margin-top: 14px; font-size: 14px; color: var(--muted); font-weight: 300; max-width: 500px; line-height: 1.65; }}
.stats-row {{ display: flex; gap: 2px; flex-direction: column; align-items: flex-end; }}
.stat {{ text-align: right; padding: 10px 0; border-bottom: 1px solid var(--border); width: 140px; }}
.stat:last-child {{ border-bottom: none; }}
.stat-n {{ font-family: var(--mono); font-size: 28px; font-weight: 500; }}
.stat-label {{ font-family: var(--mono); font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-top: 2px; }}

main {{ max-width: 1360px; margin: 0 auto; padding: 0 56px 80px; }}
section {{ margin-top: 64px; }}
.section-header {{ display: flex; align-items: baseline; gap: 16px; margin-bottom: 28px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }}
.section-num {{ font-family: var(--mono); font-size: 10px; color: var(--red); letter-spacing: 0.1em; }}
h2 {{ font-family: var(--serif); font-size: 24px; font-weight: 400; }}
.section-desc {{ font-size: 13px; color: var(--muted); margin-top: -16px; margin-bottom: 24px; line-height: 1.6; }}

/* FILM SIMILARITY MAP */
.filmmap-layout {{ display: flex; gap: 16px; align-items: flex-start; }}
.filmmap-wrap {{
  background: var(--bg2); border: 1px solid var(--border);
  position: relative; border-radius: 2px; overflow: hidden; flex: 1; min-width: 0;
}}
#filmmap-svg {{ width: 100%; height: 600px; display: block; }}
.filmmap-controls {{
  display: flex; align-items: center; gap: 20px;
  padding: 12px 16px; border-top: 1px solid var(--border);
  background: var(--bg2); flex-wrap: wrap;
}}
.filmmap-toggle {{ display: flex; border: 1px solid var(--border); border-radius: 2px; overflow: hidden; }}
.filmmap-toggle-btn {{
  font-family: var(--mono); font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase;
  background: var(--bg2); color: var(--muted); border: none; padding: 7px 14px; cursor: pointer;
}}
.filmmap-toggle-btn.active {{ background: var(--gold); color: var(--bg); }}
.filmmap-toggle-btn:not(.active):hover {{ background: var(--bg3); color: var(--text); }}
.filmmap-legend {{ display: flex; gap: 14px; margin-left: auto; align-items: center; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 9px; color: var(--muted); }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.quality-panel {{
  width: 260px; flex-shrink: 0;
  background: var(--bg2); border: 1px solid var(--border);
  padding: 16px; font-family: var(--mono);
}}
.quality-title {{ font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
.quality-stat {{ margin-bottom: 16px; }}
.quality-stat-n {{ font-size: 22px; color: var(--gold); font-weight: 500; line-height: 1; }}
.quality-stat-label {{ font-size: 9px; color: var(--muted); margin-top: 4px; letter-spacing: 0.04em; }}
.quality-verdict {{
  font-size: 11px; line-height: 1.6; padding: 10px 12px; margin-top: 14px;
  border-left: 2px solid var(--red); background: var(--bg3); color: var(--text);
}}
.quality-verdict.ok {{ border-left-color: var(--gold); }}
.sil-bars {{ margin-top: 14px; }}
.sil-row {{ display: grid; grid-template-columns: 24px 1fr 40px; gap: 6px; align-items: center; margin-bottom: 4px; font-size: 9px; color: var(--muted); }}
.sil-track {{ height: 10px; background: var(--bg3); border-radius: 1px; overflow: hidden; }}
.sil-fill {{ height: 100%; background: var(--gold); }}
.sil-fill.best {{ background: var(--red2); }}
.sil-fill.neg {{ background: var(--muted); }}
.method-note {{
  font-family: var(--mono); font-size: 10px; color: var(--muted); line-height: 1.7;
  background: var(--bg2); border: 1px solid var(--border); padding: 12px 16px; margin-top: 16px;
}}
.method-note b {{ color: var(--gold); font-weight: 500; }}
input[type=range] {{
  -webkit-appearance: none; appearance: none;
  width: 200px; height: 2px; background: var(--border); border-radius: 1px; outline: none;
}}
input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none; appearance: none;
  width: 12px; height: 12px; background: var(--gold); border-radius: 50%; cursor: pointer;
}}

/* PAIR CARDS */
.pair-list {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; align-items: start; }}
.pair-card {{
  display: flex; flex-direction: column;
  background: var(--bg2); border: 1px solid var(--border); border-radius: 3px;
  padding: 14px 16px; transition: background 0.1s;
  min-width: 0; overflow: hidden;
}}
.pair-card:hover {{ background: var(--bg3); }}
.pair-top-row {{ display: flex; align-items: center; gap: 14px; margin-bottom: 10px; min-width: 0; }}
.pair-score {{ font-family: var(--mono); font-size: 30px; font-weight: 500; color: var(--gold); flex-shrink: 0; line-height: 1; }}
.pair-docs {{ flex: 1; min-width: 0; display: flex; flex-direction: column; align-items: center; gap: 1px; font-size: 12px; font-weight: 500; line-height: 1.3; }}
.pair-doc-name {{ max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.pair-docs .pair-vs {{ color: var(--muted); font-weight: 400; font-size: 10px; line-height: 1.4; align-self: center; }}
.pair-score-bar {{ height: 3px; background: var(--bg3); border-radius: 2px; overflow: hidden; margin-bottom: 10px; }}
.pair-score-fill {{ height: 100%; background: var(--gold); border-radius: 2px; }}
.shared-tags {{ display: flex; flex-wrap: wrap; gap: 4px; align-content: flex-start; overflow: hidden; }}
.shared-tag {{
  font-family: var(--mono); font-size: 9px;
  background: var(--bg3); border: 1px solid var(--border);
  color: var(--muted); padding: 2px 7px; border-radius: 2px;
}}
.shared-tag .n {{ color: var(--red2); margin-left: 3px; }}
@media (max-width: 900px) {{
  .pair-list {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (max-width: 560px) {{
  .pair-list {{ grid-template-columns: 1fr; }}
}}

/* THEME BREADTH */
.breadth-list {{ display: flex; flex-direction: column; gap: 8px; }}
.breadth-row {{ display: grid; grid-template-columns: 220px 1fr; gap: 12px; align-items: center; }}
.breadth-label {{ font-size: 13px; color: var(--text); line-height: 1.3; }}
.breadth-bar-track {{ height: 20px; background: var(--bg2); border-radius: 2px; overflow: hidden; }}
.breadth-bar-fill {{
  height: 100%; background: var(--gold); border-radius: 2px;
  display: flex; align-items: center; justify-content: flex-end; padding-right: 8px;
  min-width: fit-content;
}}
.breadth-bar-fill span {{ font-family: var(--mono); font-size: 10px; color: #fff; font-weight: 500; white-space: nowrap; text-shadow: 0 1px 2px rgba(0,0,0,0.55); }}
@media (max-width: 640px) {{
  .breadth-row {{ grid-template-columns: 1fr; gap: 4px; }}
}}

/* CATEGORY STACKED BARS */
.cat-stack-list {{ display: flex; flex-direction: column; gap: 18px; }}
.cat-stack-row {{ display: grid; grid-template-columns: 150px 1fr; gap: 16px; align-items: center; }}
.cat-stack-name {{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.04em; color: var(--text); }}
.cat-stack-sub {{ font-family: var(--mono); font-size: 9px; color: var(--muted); margin-top: 2px; }}
.cat-stack-bar {{ display: flex; height: 28px; border-radius: 2px; overflow: hidden; background: var(--bg2); border: 1px solid var(--border); }}
.cat-stack-seg {{ height: 100%; cursor: pointer; transition: filter 0.12s; }}
.cat-stack-seg:hover {{ filter: brightness(1.25); }}
@media (max-width: 640px) {{
  .cat-stack-row {{ grid-template-columns: 1fr; gap: 6px; }}
}}

/* FILMS TOGGLE */
.films-toggle {{ display: flex; border: 1px solid var(--border); border-radius: 2px; overflow: hidden; width: fit-content; margin-bottom: 20px; }}
.films-toggle-btn {{
  font-family: var(--mono); font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase;
  background: var(--bg2); color: var(--muted); border: none; padding: 8px 16px; cursor: pointer;
}}
.films-toggle-btn.active {{ background: var(--gold); color: var(--bg); }}
.films-toggle-btn:not(.active):hover {{ background: var(--bg3); color: var(--text); }}

/* DOC CARDS (pie charts) */
.doc-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); }}
.doc-card {{ background: var(--bg2); padding: 18px; cursor: default; display: flex; flex-direction: column; align-items: center; }}
.doc-card:hover {{ background: var(--bg3); }}
.doc-name {{ font-size: 13px; font-weight: 500; margin-bottom: 14px; line-height: 1.3; text-align: center; }}
.doc-pie-wrap {{ position: relative; width: 168px; height: 168px; }}
.doc-pie-wrap svg {{ width: 100%; height: 100%; }}
.doc-pie-slice {{ stroke: var(--bg2); stroke-width: 1.5; cursor: pointer; transition: opacity 0.12s; transform-origin: 84px 84px; }}
.doc-pie-slice:hover {{ opacity: 0.8; }}
.doc-pie-slice.dimmed {{ opacity: 0.25; }}
.doc-pie-center {{
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  text-align: center; pointer-events: none;
  width: 56px; height: 56px; border-radius: 50%; background: var(--bg2);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}}
.doc-pie-center .n {{ font-family: var(--mono); font-size: 16px; font-weight: 500; color: var(--text); }}
.doc-pie-center .label {{ font-family: var(--mono); font-size: 8px; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px; }}

#tooltip {{
  position: fixed; display: none; background: var(--bg2); border: 1px solid var(--border);
  padding: 10px 14px; font-family: var(--mono); font-size: 10px;
  color: var(--text); pointer-events: none; z-index: 999; max-width: 220px; line-height: 1.5;
}}

footer {{
  border-top: 1px solid var(--border); padding: 20px 56px;
  font-family: var(--mono); font-size: 9px; color: var(--muted);
  display: flex; justify-content: space-between; letter-spacing: 0.08em; text-transform: uppercase;
}}
@media (max-width: 768px) {{
  nav, .hero, main, footer {{ padding-left: 20px; padding-right: 20px; }}
  .hero {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div id="tooltip"></div>

<nav>
  <span class="nav-brand">Theme Analysis</span>
  <a href="#filmmap">Film Map</a>
  <a href="#chunkmap">Chunk Map</a>
{character_nav_link}
  <a href="#pairs">Top Pairs</a>
  <a href="#breadth">Theme Breadth</a>
  <a href="#categories">Who Talks About What</a>
  <a href="#films">Films</a>
</nav>

<div class="hero">
  <div>
    <div class="hero-eyebrow">BERTopic · Cosine Similarity · {strategy_label}</div>
    <h1>How Gun Violence Documentaries<br><em>Connect to Each Other</em></h1>
    <p class="hero-sub">Across {data["n_docs"]} films and {data["n_chunks"]:,} transcript segments, thematic similarity reveals which documentaries share the deepest narrative common ground — and which stand apart.</p>
  </div>
  <div class="stats-row">
    <div class="stat"><div class="stat-n">{data["n_docs"]}</div><div class="stat-label">Documentaries</div></div>
    <div class="stat"><div class="stat-n">{data["n_topics"]}</div><div class="stat-label">Themes found</div></div>
    <div class="stat"><div class="stat-n">{data["n_chunks"]:,}</div><div class="stat-label">Segments</div></div>
  </div>
</div>

<main>

    <!-- 01 FILM MAP — BY TOPIC -->
    <section id="filmmap-topic">
      <div class="section-header">
        <span class="section-num">01</span>
        <h2>Film Map — By Theme</h2>
      </div>
      <p class="section-desc">
        Each point represents a single documentary film. Its position is determined by averaging the high-dimensional text embeddings of all its transcript chunks (centroid pooling), which captures the film's overall thematic profile; UMAP then compresses that down into flat 2D coordinates for display. <strong>Color</strong> encodes each film's dominant BERTopic theme. <strong>Size</strong> reflects how dominant that theme actually is — a large dot means one theme accounts for most of the film's classified chunks; a small dot means the film's themes are more evenly spread, so its "dominant" label is a weaker summary of the whole film.
      </p>
      <div class="filmmap-wrap" style="width: 100%;">
        <svg id="filmmap-topic-svg"></svg>
        <div class="filmmap-controls" style="padding: 16px; background: var(--bg2); display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 12px;">
          <div class="filmmap-legend" id="filmmap-topic-legend" style="margin-left: 0; display: flex; gap: 16px; flex-wrap: wrap;"></div>
        </div>
      </div>
    </section>

    <!-- 01B FILM MAP — BY SPEAKER CATEGORY -->
    <section id="filmmap-speaker">
      <div class="section-header">
        <span class="section-num">01B</span>
        <h2>Film Map — By Speaker Category</h2>
      </div>
      <p class="section-desc">
        Same layout as the map above (identical film positions). <strong>Color</strong> encodes each film's dominant speaker category. <strong>Size</strong> reflects how dominant that category is — a large dot means one speaker category accounts for most of the film's classified chunks; a small dot means the film's speakers are more evenly spread across categories.
      </p>
      <div class="filmmap-wrap" style="width: 100%;">
        <svg id="filmmap-speaker-svg"></svg>
        <div class="filmmap-controls" style="padding: 16px; background: var(--bg2); display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 12px;">
          <div class="filmmap-legend" id="filmmap-speaker-legend" style="margin-left: 0; display: flex; gap: 16px; flex-wrap: wrap;"></div>
        </div>
      </div>
    </section>

    <!-- 01C FILM MAP — BY PRODUCTION & DISTRIBUTION -->
    <section id="filmmap-production">
      <div class="section-header">
        <span class="section-num">01C</span>
        <h2>Film Map — By Production &amp; Distribution</h2>
      </div>
      <p class="section-desc">
        Same layout again. <strong>Color</strong> encodes streaming/distribution platform (Major Platform, Free Streaming, YouTube, Buy/Rent), and the tooltip also shows whether the film is Studio/Network-backed or Independent. Films with no reliable classification (fuzzy-match miss, or filtered out upstream for low confidence) are shown in a neutral gray rather than guessed — that's a genuine "unknown," not "assumed independent." Size is uniform on this map since production/distribution has no natural "dominance" measure the way theme or speaker share does.
      </p>
      <div class="filmmap-wrap" style="width: 100%;">
        <svg id="filmmap-production-svg"></svg>
        <div class="filmmap-controls" style="padding: 16px; background: var(--bg2); display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 12px;">
          <div class="filmmap-legend" id="filmmap-production-legend" style="margin-left: 0; display: flex; gap: 16px; flex-wrap: wrap;"></div>
        </div>
      </div>
    </section>

    <!-- 01D CHUNK-LEVEL MAP -->
    <section id="chunkmap">
      <div class="section-header">
        <span class="section-num">01D</span>
        <h2>Chunk-Level Map</h2>
      </div>
      <p class="section-desc">
        Same idea as the film map above, but with no averaging: every point here is a single
        transcript chunk plotted from its own raw embedding via a fresh, independent UMAP fit —
        not a zoomed-in view of the film centroids above, which have no notion of individual
        chunks. The film map can only ever show structure that survives averaging every chunk in
        a film together; this shows whether individual moments of speech actually separate by
        speaker category before any aggregation happens at all. Colored by speaker category —
        type a film name below to highlight just that film's chunks against the rest.
        <span id="chunkmap-sample-note" style="display:none; color: var(--muted);"></span>
      </p>
      <div class="filmmap-wrap" style="width: 100%; position: relative;">
        <canvas id="chunkmap-canvas" style="width: 100%; height: 600px; display: block;"></canvas>
        <div class="filmmap-controls" style="padding: 16px; background: var(--bg2); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
          <input id="chunkmap-film-filter" type="text" placeholder="Highlight a film…"
                 style="background: var(--bg3); border: 1px solid var(--border); color: var(--text);
                        font-family: var(--mono); font-size: 11px; padding: 6px 10px; border-radius: 2px; width: 220px;">
          <div class="filmmap-legend" id="chunkmap-legend" style="margin-left: 0; display: flex; gap: 16px; flex-wrap: wrap;"></div>
        </div>
      </div>
    </section>
{character_section_html}
  <!-- 02 TOP PAIRS -->
  <section id="pairs">
    <div class="section-header">
      <span class="section-num">02</span>
      <h2>Strongest Connections</h2>
    </div>
    <p class="section-desc">Top 20 pairs ranked by cosine similarity, with the shared themes that drive each score.</p>
    <div class="pair-list" id="pair-list"></div>
  </section>

  <!-- 03 THEME BREADTH -->
  <section id="breadth">
    <div class="section-header">
      <span class="section-num">03</span>
      <h2>Theme Breadth</h2>
    </div>
    <p class="section-desc">How many of the {data["n_docs"]} films each theme appears in — corpus-wide reach, not depth within any one film.</p>
    <div class="breadth-list" id="breadth-list"></div>
  </section>

  <!-- 04 CATEGORY x TOPIC -->
  <section id="categories">
    <div class="section-header">
      <span class="section-num">04</span>
      <h2>Who Talks About What</h2>
    </div>
    <p class="section-desc">Each row is a speaker category. The full bar is 100% of that category's classified segments — segment width is a theme's share within that category, so a long segment means that category leans heavily on that theme. Hover a segment for the theme and exact share.</p>
    <div class="cat-stack-list" id="cat-stack-list"></div>
  </section>

  <!-- 05 FILMS -->
  <section id="films">
    <div class="section-header">
      <span class="section-num">05</span>
      <h2>Films &amp; Primary <span id="films-mode-label">Themes</span></h2>
    </div>
    <p class="section-desc" id="films-desc">Top themes per film, sized by share of that film's own classified segments — so short and long films compare fairly.</p>
    <div class="films-toggle" id="films-toggle">
      <button class="films-toggle-btn active" data-mode="theme">By theme</button>
      <button class="films-toggle-btn" data-mode="category">By speaker category</button>
    </div>
    <div class="doc-grid" id="doc-grid"></div>
  </section>

</main>

<footer>
  <span>WhisperX → BERTopic (all-mpnet-base-v2 + HDBSCAN) → Gemma 4 labels</span>
  <span>{data["n_docs"]} films · {data["n_topics"]} themes · {data["n_chunks"]:,} segments</span>
</footer>

<script>
const DATA = {json.dumps(data, indent=2)};

// ── TOOLTIP ───────────────────────────────────────────────────────────────
const tooltip = document.getElementById('tooltip');
function showTip(e, html) {{ tooltip.innerHTML = html; tooltip.style.display = 'block'; moveTip(e); }}
function moveTip(e) {{
  const x = e.clientX + 14, y = e.clientY - 10;
  tooltip.style.left = Math.min(x, window.innerWidth - 240) + 'px';
  tooltip.style.top = y + 'px';
}}
function hideTip() {{ tooltip.style.display = 'none'; }}
document.addEventListener('mousemove', moveTip);

function cleanName(s) {{ return s.replace(/_Transcript$|_Transcript\\.docx$/, '').replace(/_/g, ' '); }}

// ── GLOBAL THEME COLOR MAP ───────────────────────────────────────────────
// One stable, muted color per theme, shared by every chart on the page
// (pies, bars, tags) so the same theme always reads as the same color —
// desaturated and darkened to sit alongside the page's dusty red/gold palette.
// Hue comes from DATA.theme_hue when available (a PCA projection of each
// theme's label embedding, so semantically related themes land on nearby
// hues) and falls back to evenly-spaced hues for any theme missing one.
const THEME_COLOR = (function() {{
  const allThemes = (DATA.topic_summary || []).map(t => t.label).sort();
  const n = allThemes.length || 1;
  const semanticHue = DATA.theme_hue || {{}};
  const map = {{}};
  allThemes.forEach((label, i) => {{
    const hue = semanticHue[label] !== undefined ? semanticHue[label] : Math.round((i / n) * 320);
    map[label] = `hsl(${{hue}}, 30%, 40%)`;
  }});
  map['Other'] = 'hsl(0, 0%, 32%)';
  return map;
}})();
function themeColor(label) {{ return THEME_COLOR[label] || '#888'; }}

// ── SPEAKER CATEGORY COLOR MAP ───────────────────────────────────────────
// Separate from THEME_COLOR since categories (8-9 fixed buckets) are a
// different dimension than themes. Evenly spaced is fine at this scale —
// there's no long tail and no semantic-clustering need for so few buckets.
const CATEGORY_COLOR = (function() {{
  const cats = (DATA.category_breakdown?.categories || []).slice().sort();
  const n = cats.length || 1;
  const map = {{}};
  cats.forEach((label, i) => {{
    const hue = Math.round((i / n) * 320);
    map[label] = `hsl(${{hue}}, 30%, 40%)`;
  }});
  map['OTHER'] = 'hsl(0, 0%, 32%)';
  return map;
}})();
function categoryColor(label) {{ return CATEGORY_COLOR[label] || '#888'; }}

// ── FILM MAPS (D3 scatter, one per encoding) ────────────────────────────────
// All three maps below share the same x/y film positions (from the one UMAP
// fit in film_similarity) and the same shell logic. Rather than one chart
// with toggled shape/color/border channels, each map is now a single,
// purely color-based encoding — makeFilmMap() is the shared renderer;
// each call below only supplies what differs (svg/legend ids, color
// function, size function, legend entries, tooltip line).
const STREAMING_COLOR = {{
  'Major Platform': 'hsl(265, 55%, 62%)',
  'Free Streaming': 'hsl(150, 45%, 50%)',
  'YouTube': 'hsl(0, 60%, 55%)',
  'Buy/Rent': 'hsl(45, 75%, 55%)',
}};
const NEUTRAL_FILL = 'hsl(35, 12%, 46%)';

function makeFilmMap(opts) {{
  try {{
    const svgEl = document.getElementById(opts.svgId);
    if (!svgEl) return;
    const fsim = DATA.film_similarity;
    const films = fsim.films;

    const svg = d3.select(svgEl);
    const container = svgEl.parentElement;
    const W = container.offsetWidth || 1100, H = 600;
    svg.attr('viewBox', `0 0 ${{W}} ${{H}}`);

    const PAD = 60;
    const xExtent = d3.extent(films, d => d.x);
    const yExtent = d3.extent(films, d => d.y);
    const xScale = d3.scaleLinear().domain(xExtent).range([PAD, W - PAD]).nice();
    const yScale = d3.scaleLinear().domain(yExtent).range([H - PAD, PAD]).nice();

    // Size channel: either a metric-driven sqrt scale (topic/speaker maps,
    // where "dominance" is a real, meaningful percentage) or a fixed radius
    // (production map, where there's no analogous per-film metric).
    let radiusFor;
    if (opts.sizeMetric) {{
      const extent = d3.extent(films, opts.sizeMetric);
      const rScale = d3.scaleSqrt()
        .domain(extent[0] === extent[1] ? [0, extent[1] || 1] : extent)
        .range([6, 20]);
      radiusFor = d => rScale(opts.sizeMetric(d) || 0);
    }} else {{
      radiusFor = () => 11;
    }}

    const gGrid = svg.append('g');
    gGrid.selectAll('line.h').data(yScale.ticks(5)).enter().append('line')
      .attr('x1', PAD).attr('x2', W - PAD).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
      .attr('stroke', '#1f1f1f').attr('stroke-width', 1);
    gGrid.selectAll('line.v').data(xScale.ticks(5)).enter().append('line')
      .attr('y1', PAD).attr('y2', H - PAD).attr('x1', d => xScale(d)).attr('x2', d => xScale(d))
      .attr('stroke', '#1f1f1f').attr('stroke-width', 1);

    const gPoints = svg.append('g');

    gPoints.selectAll('circle.film-node').data(films, d => d.doc).join('circle')
      .attr('class', 'film-node')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', radiusFor)
      .attr('fill', opts.colorFor)
      .attr('fill-opacity', 0.85)
      .attr('stroke', '#e6e2d8')
      .attr('stroke-width', 0.75)
      .style('cursor', 'pointer')
      .on('mouseenter', function(event, d) {{
        d3.select(this).attr('stroke-width', 2);
        showTip(event, opts.tooltipFor(d));
      }})
      .on('mouseleave', function() {{
        d3.select(this).attr('stroke-width', 0.75);
        hideTip();
      }});

    gPoints.selectAll('text').data(films, d => d.doc).join('text')
      .attr('x', d => xScale(d.x))
      .attr('y', d => yScale(d.y) - radiusFor(d) - 6)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'IBM Plex Mono, monospace')
      .attr('font-size', '9px')
      .attr('fill', '#9a9088')
      .attr('pointer-events', 'none')
      .text(d => {{
        const name = cleanName(d.doc);
        return name.length > 16 ? name.slice(0, 14) + '…' : name;
      }});

    const legend = document.getElementById(opts.legendId);
    if (legend) legend.innerHTML = opts.legendHtml(films);

    svg.call(d3.zoom().scaleExtent([0.5, 4]).on('zoom', e => {{
      gPoints.attr('transform', e.transform);
    }}));
  }} catch (err) {{
    console.error(`Film map (${{opts.svgId}}) failed to render:`, err);
  }}
}}

function legendFromCounts(items, colorFn, maxShown) {{
  const counts = {{}};
  items.forEach(label => {{ if (label) counts[label] = (counts[label] || 0) + 1; }});
  const ranked = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  let html = ranked.slice(0, maxShown).map(label =>
    `<div class="legend-item"><div class="legend-dot" style="background:${{colorFn(label)}}"></div>${{label}}</div>`
  ).join('');
  if (ranked.length > maxShown) {{
    html += `<div class="legend-item" style="color:var(--muted)">+${{ranked.length - maxShown}} more (see tooltip)</div>`;
  }}
  return html;
}}

// ── 01: by theme ─────────────────────────────────────────────────────────
makeFilmMap({{
  svgId: 'filmmap-topic-svg',
  legendId: 'filmmap-topic-legend',
  sizeMetric: d => d.dominant_theme_pct,
  colorFor: d => d.dominant_theme ? themeColor(d.dominant_theme) : themeColor('Other'),
  tooltipFor: d => {{
    const themeLine = d.dominant_theme
      ? `Dominant theme: ${{d.dominant_theme}} (${{d.dominant_theme_pct}}%)`
      : 'Dominant theme: —';
    return `<b>${{cleanName(d.doc)}}</b><br>${{d.chunks}} segments<br>${{themeLine}}`;
  }},
  legendHtml: films => legendFromCounts(films.map(d => d.dominant_theme), themeColor, 12),
}});

// ── 01B: by speaker category ──────────────────────────────────────────────
makeFilmMap({{
  svgId: 'filmmap-speaker-svg',
  legendId: 'filmmap-speaker-legend',
  sizeMetric: d => d.dominant_category_pct,
  colorFor: d => categoryColor(d.dominant_category),
  tooltipFor: d => `<b>${{cleanName(d.doc)}}</b><br>${{d.chunks}} segments<br>` +
    `Dominant speaker: ${{d.dominant_category}} (${{d.dominant_category_pct}}%)`,
  legendHtml: films => legendFromCounts(films.map(d => d.dominant_category), categoryColor, 12),
}});

// ── 01C: by production & distribution ─────────────────────────────────────
makeFilmMap({{
  svgId: 'filmmap-production-svg',
  legendId: 'filmmap-production-legend',
  sizeMetric: null,   // uniform size -- no per-film "dominance" metric here
  colorFor: d => d.streaming_category ? (STREAMING_COLOR[d.streaming_category] || NEUTRAL_FILL) : NEUTRAL_FILL,
  tooltipFor: d => {{
    const prod = d.production_type ? `Production: ${{d.production_type}}` : 'Production: unclassified (low confidence excluded)';
    const stream = d.streaming_category ? `Streaming: ${{d.streaming_category}}` : 'Streaming: unknown';
    return `<b>${{cleanName(d.doc)}}</b><br>${{d.chunks}} segments<br>${{prod}}<br>${{stream}}`;
  }},
  legendHtml: films => {{
    let html = legendFromCounts(films.map(d => d.streaming_category), l => STREAMING_COLOR[l] || NEUTRAL_FILL, 8);
    html += `<div class="legend-item"><div class="legend-dot" style="background:${{NEUTRAL_FILL}}"></div>Unclassified</div>`;
    return html;
  }},
}});

// ── CHUNK-LEVEL MAP (canvas scatter) ────────────────────────────────────────
// One point per chunk, no aggregation. Rendered on canvas rather than one
// SVG element per point since chunk counts (unlike film counts) can run into
// the thousands depending on chunking approach, and canvas stays responsive
// at that scale where per-node SVG DOM elements would not.
try {{
(function() {{
  const scatter = DATA.chunk_scatter;
  if (!scatter || !scatter.points || !scatter.points.length) return;
  const points = scatter.points;

  const noteEl = document.getElementById('chunkmap-sample-note');
  if (scatter.sampled && noteEl) {{
    noteEl.style.display = 'inline';
    noteEl.textContent = ` Showing a random sample of ${{scatter.n_shown.toLocaleString()}} of ${{scatter.n_total_chunks.toLocaleString()}} total chunks for render performance.`;
  }}

  const canvas = document.getElementById('chunkmap-canvas');
  const ctx = canvas.getContext('2d');
  const wrap = canvas.parentElement;
  const W = wrap.offsetWidth || 1100, H = 600;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(dpr, dpr);

  const PAD = 30;
  const xExtent = d3.extent(points, d => d.x);
  const yExtent = d3.extent(points, d => d.y);
  const xScale = d3.scaleLinear().domain(xExtent).range([PAD, W - PAD]).nice();
  const yScale = d3.scaleLinear().domain(yExtent).range([H - PAD, PAD]).nice();

  // Precompute pixel positions once -- reused every redraw (highlight toggle)
  // and by the quadtree used for hover hit-testing.
  points.forEach(d => {{ d.px = xScale(d.x); d.py = yScale(d.y); }});

  const quadtree = d3.quadtree().x(d => d.px).y(d => d.py).addAll(points);

  const cats = Array.from(new Set(points.map(d => d.category))).sort();
  const legend = document.getElementById('chunkmap-legend');
  legend.innerHTML = '';
  cats.forEach(cat => {{
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.style.cursor = 'pointer';
    item.dataset.cat = cat;
    item.innerHTML = `<div class="legend-dot" style="background:${{categoryColor(cat)}}"></div>${{cat}}`;
    legend.appendChild(item);
  }});

  const R = 2.4;
  let highlightDoc = null;
  let highlightCategory = null;

  function isFocused(d) {{
    if (highlightDoc && d.doc !== highlightDoc) return false;
    if (highlightCategory && d.category !== highlightCategory) return false;
    return true;
  }}

  function draw() {{
    ctx.clearRect(0, 0, W, H);
    const anyFilter = highlightDoc || highlightCategory;
    // Dim pass first (so focused points draw on top, un-occluded), then the
    // focused points at full opacity -- shared by both the film-highlight
    // text box and the click-a-legend-item-to-isolate-a-category behavior
    // below, since either (or both together) just narrows what counts as
    // "focused."
    points.forEach(d => {{
      if (anyFilter && !isFocused(d)) {{
        ctx.beginPath();
        ctx.arc(d.px, d.py, R, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(120,115,105,0.15)';
        ctx.fill();
      }}
    }});
    points.forEach(d => {{
      if (anyFilter && !isFocused(d)) return;
      const isHighlightedDoc = highlightDoc && d.doc === highlightDoc;
      ctx.beginPath();
      ctx.arc(d.px, d.py, isHighlightedDoc ? R * 1.8 : R, 0, Math.PI * 2);
      ctx.fillStyle = categoryColor(d.category);
      ctx.globalAlpha = isHighlightedDoc ? 1 : 0.7;
      ctx.fill();
    }});
    ctx.globalAlpha = 1;
  }}

  draw();

  legend.querySelectorAll('.legend-item').forEach(item => {{
    item.addEventListener('click', () => {{
      const cat = item.dataset.cat;
      highlightCategory = (highlightCategory === cat) ? null : cat;
      legend.querySelectorAll('.legend-item').forEach(i => {{
        i.style.opacity = (!highlightCategory || i.dataset.cat === highlightCategory) ? '1' : '0.35';
      }});
      draw();
    }});
  }});

  canvas.addEventListener('mousemove', (e) => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const found = quadtree.find(mx, my, 12);
    if (found) {{
      showTip(e, `<b>${{cleanName(found.doc)}}</b><br>${{found.category}}<br><span style="color:var(--muted)">${{found.snippet}}</span>`);
      canvas.style.cursor = 'pointer';
    }} else {{
      hideTip();
      canvas.style.cursor = 'default';
    }}
  }});
  canvas.addEventListener('mouseleave', hideTip);

  const filterInput = document.getElementById('chunkmap-film-filter');
  const allDocs = Array.from(new Set(points.map(d => d.doc)));
  if (filterInput) {{
    filterInput.addEventListener('input', () => {{
      const q = filterInput.value.trim().toLowerCase();
      highlightDoc = q ? (allDocs.find(doc => cleanName(doc).toLowerCase().includes(q)) || null) : null;
      draw();
    }});
  }}
}})();
}} catch (err) {{
  console.error('Chunk-level map failed to render:', err);
}}
{character_map_js}
// ── TOP PAIRS (cards) ──────────────────────────────────────────────────────
(function() {{
  const list = document.getElementById('pair-list');
  const maxScore = DATA.top_pairs[0]?.score || 1;
  DATA.top_pairs.slice(0, 20).forEach(p => {{
    const card = document.createElement('div'); card.className = 'pair-card';
    const pct = (p.score / maxScore * 100).toFixed(1);
    const tags = p.shared.slice(0, 4).map(s =>
      `<span class="shared-tag" style="border-left:3px solid ${{themeColor(s.label)}}">${{s.label}}<span class="n">${{s.total}}</span></span>`
    ).join('');
    card.innerHTML = `
      <div class="pair-top-row">
        <div class="pair-score">${{p.score.toFixed(2)}}</div>
        <div class="pair-docs">
          <span class="pair-doc-name">${{cleanName(p.doc1)}}</span>
          <span class="pair-vs">&harr;</span>
          <span class="pair-doc-name">${{cleanName(p.doc2)}}</span>
        </div>
      </div>
      <div class="pair-score-bar"><div class="pair-score-fill" style="width:${{pct}}%"></div></div>
      <div class="shared-tags">${{tags}}</div>`;
    list.appendChild(card);
  }});
}})();

// ── THEME BREADTH ──────────────────────────────────────────────────────────
(function() {{
  const list = document.getElementById('breadth-list');
  const sorted = [...DATA.topic_summary].sort((a, b) => b.doc_count - a.doc_count);
  const maxCount = sorted[0]?.doc_count || 1;
  sorted.forEach(t => {{
    const row = document.createElement('div'); row.className = 'breadth-row';
    const widthPct = (t.doc_count / maxCount * 100).toFixed(1);
    row.innerHTML = `
      <div class="breadth-label">${{t.label}}</div>
      <div class="breadth-bar-track">
        <div class="breadth-bar-fill" style="width:${{widthPct}}%; background:${{themeColor(t.label)}}">
          <span>${{t.doc_count}} film${{t.doc_count === 1 ? '' : 's'}}</span>
        </div>
      </div>`;
    list.appendChild(row);
  }});
}})();

// ── CATEGORY STACKED BARS ───────────────────────────────────────────────────
(function() {{
  const breakdown = DATA.category_breakdown;
  if (!breakdown) return;

  const list = document.getElementById('cat-stack-list');

  breakdown.categories.forEach(cat => {{
    const themes = breakdown.by_category[cat] || [];
    const row = document.createElement('div'); row.className = 'cat-stack-row';

    const segs = themes.map(t =>
      `<div class="cat-stack-seg" style="width:${{t.pct}}%; background:${{themeColor(t.label)}}"
            data-label="${{t.label}}" data-pct="${{t.pct}}" data-count="${{t.count}}"></div>`
    ).join('');

    row.innerHTML = `
      <div>
        <div class="cat-stack-name">${{cat}}</div>
        <div class="cat-stack-sub">${{themes.length}} theme${{themes.length === 1 ? '' : 's'}}</div>
      </div>
      <div class="cat-stack-bar">${{segs}}</div>`;
    list.appendChild(row);
  }});

  list.querySelectorAll('.cat-stack-seg').forEach(seg => {{
    seg.addEventListener('mouseenter', (e) => {{
      const {{ label, pct, count }} = seg.dataset;
      showTip(e, `<b>${{label}}</b><br><span style="color:${{themeColor(label)}}">${{pct}}%</span> of segments <span style="color:var(--muted)">(${{count}})</span>`);
    }});
    seg.addEventListener('mouseleave', hideTip);
  }});
}})();

// ── FILMS — PIE CHARTS ──────────────────────────────────────────────────────
(function() {{
  const grid = document.getElementById('doc-grid');
  if (!grid) return;

  const R = 84, CX = 84, CY = 84;
  let mode = 'theme';

  function arcPath(startAngle, endAngle, r, cx, cy) {{
    // angles in radians, 0 = top, clockwise
    // A full 360° sweep (single category/theme at 100%) can't be drawn as one
    // SVG arc command — its start and end points are identical, so the path
    // is zero-length and invisible. Split it into two half-circle arcs instead.
    const full = (endAngle - startAngle) >= Math.PI * 2 - 0.0001;
    if (full) {{
      const midAngle = startAngle + Math.PI;
      const x1 = cx + r * Math.sin(startAngle), y1 = cy - r * Math.cos(startAngle);
      const xm = cx + r * Math.sin(midAngle),   ym = cy - r * Math.cos(midAngle);
      return `M${{cx}},${{cy}} L${{x1.toFixed(2)}},${{y1.toFixed(2)}} A${{r}},${{r}} 0 1 1 ${{xm.toFixed(2)}},${{ym.toFixed(2)}} A${{r}},${{r}} 0 1 1 ${{x1.toFixed(2)}},${{y1.toFixed(2)}} Z`;
    }}
    const x1 = cx + r * Math.sin(startAngle), y1 = cy - r * Math.cos(startAngle);
    const x2 = cx + r * Math.sin(endAngle),   y2 = cy - r * Math.cos(endAngle);
    const large = (endAngle - startAngle) > Math.PI ? 1 : 0;
    return `M${{cx}},${{cy}} L${{x1.toFixed(2)}},${{y1.toFixed(2)}} A${{r}},${{r}} 0 ${{large}} 1 ${{x2.toFixed(2)}},${{y2.toFixed(2)}} Z`;
  }}

  function sliceData(doc) {{
    if (mode === 'category') {{
      // doc_categories is exhaustive (every chunk lands in a named bucket or
      // OTHER), so it always sums to ~100% already — no extra slice needed.
      return {{ items: DATA.doc_categories[doc] || [], color: categoryColor, unit: 'category' }};
    }}
    // doc_topics already includes a fully-formed 'Other' entry (with its own
    // count/pct/n_themes) for whatever fell under the 2% threshold server-side
    // — nothing to synthesize here.
    return {{ items: DATA.doc_topics[doc] || [], color: themeColor, unit: 'theme' }};
  }}

  function render() {{
    grid.innerHTML = '';
    DATA.docs.forEach(doc => {{
      const {{ items, color, unit }} = sliceData(doc);
      // Total count of distinct things this film covers — slices shown
      // individually, plus however many got pooled into Other (n_themes),
      // so the center number reflects every theme/category present, not
      // just how many wedges got drawn.
      const otherEntry = items.find(s => s.label === 'Other');
      const realCount = items.filter(s => s.label !== 'Other').length + (otherEntry?.n_themes || 0);

      const card = document.createElement('div'); card.className = 'doc-card';
      const svgNS = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(svgNS, 'svg');
      svg.setAttribute('viewBox', `0 0 ${{CX * 2}} ${{CY * 2}}`);

      let angle = 0;
      items.forEach(s => {{
        const frac = Math.max(0, s.pct) / 100;
        const sweep = frac * Math.PI * 2;
        const path = document.createElementNS(svgNS, 'path');
        path.setAttribute('d', arcPath(angle, angle + sweep, R, CX, CY));
        path.setAttribute('fill', color(s.label));
        path.setAttribute('class', 'doc-pie-slice');
        path.addEventListener('mouseenter', (e) => {{
          const detail = s.label === 'Other' && s.n_themes
            ? `<br><span style="color:var(--muted)">${{s.n_themes}} theme${{s.n_themes === 1 ? '' : 's'}} under 2% each</span>`
            : '';
          showTip(e, `<b>${{s.label}}</b><br><span style="color:${{color(s.label)}}">${{s.pct.toFixed(1)}}%</span>${{detail}}`);
        }});
        path.addEventListener('mouseleave', hideTip);
        svg.appendChild(path);
        angle += sweep;
      }});

      const wrap = document.createElement('div'); wrap.className = 'doc-pie-wrap';
      wrap.appendChild(svg);
      const center = document.createElement('div'); center.className = 'doc-pie-center';
      const unitPlural = realCount === 1 ? unit : (unit === 'category' ? 'categories' : `${{unit}}s`);
      center.innerHTML = `<div class="n">${{realCount}}</div><div class="label">${{unitPlural}}</div>`;
      wrap.appendChild(center);

      const name = document.createElement('div'); name.className = 'doc-name';
      name.textContent = cleanName(doc);

      card.appendChild(name);
      card.appendChild(wrap);
      grid.appendChild(card);
    }});
  }}

  const toggle = document.getElementById('films-toggle');
  const modeLabel = document.getElementById('films-mode-label');
  const desc = document.getElementById('films-desc');
  if (toggle) {{
    toggle.querySelectorAll('.films-toggle-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        mode = btn.dataset.mode;
        toggle.querySelectorAll('.films-toggle-btn').forEach(b => b.classList.toggle('active', b === btn));
        if (modeLabel) modeLabel.textContent = mode === 'category' ? 'Speaker Categories' : 'Themes';
        if (desc) desc.textContent = mode === 'category'
          ? "Speaker-category share per film, by percent of that film's own classified segments."
          : "Top themes per film, sized by share of that film's own classified segments — so short and long films compare fairly.";
        render();
      }});
    }});
  }}

  render();
}})();

</script>
</body>
</html>"""

    return html.replace('__D3_INJECT__', f'<script>{d3_js}</script>')


def export_documentary_html(
    results_df,
    topic_model,
    new_labels,
    embedding_model,
    d3_js,
    notebook_name,
    output_dir='visualizations',
    embeddings=None,
    strategy_label='Time-Window, Speaker-Metadata',
    chunks_are_single_speaker=False,
    production_type_map=None,
    streaming_category_map=None,
):
    """Full pipeline: results_df -> data payload -> rendered HTML -> file on
    disk. This is the only function approach notebooks need to call.

    Parameters mirror exactly what the old inline cell expected to find
    already sitting in notebook scope — they're explicit arguments here
    instead, so this module has no hidden dependency on notebook globals.

    Args:
        results_df: DataFrame with columns documentary, category, topic_id
            (one row per chunk). This is the one thing that's genuinely
            different between approaches — #3/#4/#5 produce this with
            different chunking logic, everything else is unchanged.
        topic_model: fitted BERTopic model (used for topic_id -> name lookup
            and get_topic_info()).
        new_labels: dict mapping topic_id -> human-readable label override.
        embedding_model: the SentenceTransformer used for clustering — reused
            here to embed topic labels for the semantic hue mapping.
        d3_js: the D3 library source as a string, to inline into the page.
        notebook_name: short identifier for the calling notebook/approach
            (e.g. 'approach_3_turn_strict'). Used verbatim as the output
            filename (`<notebook_name>.html`) so every approach's page lives
            side by side in the same output folder without collisions or
            everyone overwriting a shared 'documentary_connections.html'.
            Keep it filesystem-safe (letters, digits, `_`/`-`); no need to
            add '.html' yourself.
        output_dir: directory the HTML file gets written to. Defaults to a
            single shared 'visualizations' folder for the whole project —
            override only if you have a specific reason to write elsewhere.
        strategy_label: short name for this approach's chunking/modeling
            strategy, shown in the hero eyebrow ("BERTopic · Cosine
            Similarity · <strategy_label>"). Pass a different label per
            approach notebook — e.g. 'Turn-Strict, Speaker-Structure' for #3,
            'Turn-Merge-Coarse' for #4, 'Hybrid-Capped' for #5 — so the page
            names its own methodology correctly instead of always describing
            whichever approach this default was written for.
        chunks_are_single_speaker: pass True only for chunking strategies
            that guarantee one speaker per chunk (turn-strict and similar,
            NOT time-window) -- enables the algorithmic character-class map.
            See build_character_class_payload's docstring for why this
            can't be auto-detected and must be asserted explicitly.
        production_type_map, streaming_category_map: optional dicts (see
            build_data_payload's docstring) enabling the film map's border
            encoding -- solid/dashed border for production type, border
            color for streaming platform. Fuzzy-matched against
            results_df['documentary'], so exact key formatting doesn't need
            to match your filenames. Low-confidence classifications should
            already be filtered out of production_type_map before it's
            passed in here.

    Returns:
        The full output path of the written HTML file.
    """
    data = build_data_payload(results_df, topic_model, new_labels, embedding_model,
                               embeddings=embeddings,
                               chunks_are_single_speaker=chunks_are_single_speaker,
                               production_type_map=production_type_map,
                               streaming_category_map=streaming_category_map)
    html = render_html(data, d3_js, strategy_label=strategy_label)

    filename = f'{notebook_name}.html'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'✓  Saved: {output_path}')
    print(f'   {data["n_docs"]} films · {data["n_topics"]} themes · {data["n_chunks"]:,} segments')
    cq = data["film_similarity"]["cluster_quality"]
    print(f'   Film similarity map: {cq["hdbscan_n_clusters"]} clusters found, '
          f'{cq["hdbscan_n_noise"]} films unassigned, best silhouette {cq["best_silhouette"]}')
    if data["character_class"]:
        cc = data["character_class"]
        ccq = cc["cluster_quality"]
        print(f'   Character map: {cc["n_characters"]} speakers ({cc["n_excluded"]} excluded, too few chunks), '
              f'{ccq["hdbscan_n_clusters"]} algorithmic clusters found, '
              f'agreement with assigned categories (ARI) = {cc["agreement"]["adjusted_rand_index"]}')
    if data["top_pairs"]:
        tp = data["top_pairs"][0]
        print(f'   Top pair: {tp["doc1"]} ↔ {tp["doc2"]} ({tp["score"]:.3f})')

    return output_path


# ── A note on approach #6 ───────────────────────────────────────────────────
# #6 (per-category independent models) fits a SEPARATE BERTopic model per
# coarse_category, so there is no single shared topic space across
# categories — topic_id 3 in the BEREAVED model and topic_id 3 in the
# PROFESSIONAL model mean unrelated things. Calling export_documentary_html()
# with #6's output as if it were a single results_df will run without
# erroring, but Theme Breadth and Top Pairs will be silently meaningless:
# they assume topic_label is comparable across every row, which #6 violates
# by construction. The Film Similarity Map is the one section that's safe
# either way — it's built from raw chunk embeddings via embedding_model, not
# from topic assignments, so it doesn't care whether topic spaces are
# shared. If you get to #6, this
# module will need either (a) a separate per-category small-multiples export
# function, or (b) being called once per category with category-scoped data
# and combined into a different page layout — not a straight reuse of
# render_html() as-is.