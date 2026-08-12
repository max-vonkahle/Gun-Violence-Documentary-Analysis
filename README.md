# Documentary Gun Violence Transcript Analysis

Can NLP and machine learning surface meaningful connections between documentaries about gun violence — by analyzing what's actually said in them, and who says it?

This project builds a pipeline that takes documentary films as input, produces clean speaker-labeled transcripts, and then applies topic modeling (BERTopic) and LLM-based analysis to find recurring themes — testing, in particular, whether *who is speaking* (e.g. a bereaved family member vs. a policy expert vs. a journalist) changes what topics emerge.

## Start here

- **New to this project, or non-technical?** Read [`docs/overview.md`](docs/overview.md) — a plain-language walkthrough of what this project does and why.
- **Want the technical details on how transcripts are produced?** Read [`docs/data-acquisition.md`](docs/data-acquisition.md).
- **Want the technical details on the topic-modeling experiments?** Read [`docs/methodology.md`](docs/methodology.md) — a comparison of the modeling approaches implemented so far.

## How it works, in brief

1. **Transcript acquisition** — for each documentary, get a transcript by (in order of preference): pulling it from the hosting platform directly, extracting closed captions, or recording the audio and running it through WhisperX (transcription + speaker diarization). All three paths converge on the same output format: `[MM:SS] SPEAKER: dialogue`.
2. **Topic modeling** — the resulting transcripts are chunked by two strategies so far (fixed time windows, and speaker-turn boundaries), then fed into BERTopic to discover topics. Speaker category can be attached as metadata to either strategy to test whether it affects what's found.

## Repository structure

```
├── README.md
├── requirements.txt # exact package versions (pip freeze)
├── docs/
│ ├── overview.md # non-technical project overview
│ ├── data-acquisition.md # transcript extraction & WhisperX pipeline
│ └── methodology.md # topic-modeling approaches implemented so far
├── notebooks/
│ ├── TranscriptExtraction.ipynb # platform-specific scraping (Kanopy, Alexander Street, PBS)
│ ├── WhisperX.ipynb # audio-to-transcript fallback pipeline
│ ├── inspect_and_fix_raw_transcripts.ipynb # format inspection & normalization for non-WhisperX transcripts
│ ├── speaker_classifier.ipynb # speaker category labeling
│ └── approach_*.ipynb # one notebook per topic-modeling approach
├── raw_transcripts/ # transcripts as obtained from source, pre-normalization
│ └── docx/ # original .docx files from Alexander Street
├── transcripts/ # fully normalized transcripts ([MM:SS] SPEAKER: text)
├── docx_to_txt.py # converts raw .docx transcripts to plain .txt
├── export_html.py # shared HTML viz generator, called from each approach notebook
└── visualizations/ # HTML output, one file per approach
```

> Update this tree if folder names differ once everything's pushed — this reflects the current setup as of this writing.

## Status

This is an active, exploratory research project. The transcript-acquisition pipeline is working. Topic-modeling approaches are classified by chunking strategy; two are implemented so far (time-window and speaker-turn), with more planned — see [`docs/methodology.md`](docs/methodology.md).

## Setup & usage

```bash
pip install -r requirements.txt
```

_Add notebook-specific run instructions here (e.g. how to point a given approach notebook at a new documentary) as the pipeline stabilizes._
