# Transcript Acquisition Pipeline

This document covers how raw documentary transcripts are obtained and normalized before any topic modeling happens. Full implementation lives in [`notebooks/TranscriptExtraction.ipynb`](../notebooks/TranscriptExtraction.ipynb) and [`notebooks/WhisperX_1_.ipynb`](../notebooks/WhisperX_1_.ipynb).

## Target output format

Regardless of acquisition path, every transcript is normalized to:

```
[MM:SS] SPEAKER: Dialogue text here
```

Keeping this format constant across sources means the chunking/modeling stage (see `methodology.md`) never needs to know which extraction path a given transcript came from.

## Acquisition paths, in order of preference

### 1. Platform-native transcript scraping

Where the hosting platform displays a synchronized transcript panel, a browser-console JavaScript snippet scrapes the DOM directly rather than relying on any API. Each platform requires a different scraper, since each structures its transcript markup differently:

- **Kanopy** — timestamps and speaker labels are both embedded inside individual transcript-line elements, with the speaker name inline in brackets (e.g. `[news anchor]`). The scraper tracks the "current speaker" across lines since bracket tags only appear when the speaker changes.
- **Alexander Street** — timestamps, speaker names, and dialogue are each isolated in their own DOM elements/classes, making extraction more direct. A fallback variant handles cases where speaker names aren't a separate element and are just part of the paragraph text.
- **PBS** — the visible transcript has clean, human-verified speaker labels but *no timestamps at all*. This requires a two-part approach (see below).

Each scraper runs in the browser dev console (F12) on the transcript page itself and triggers a `.txt` download of the formatted result — no server-side scraping infrastructure involved.

### 2. Caption/subtitle extraction (`yt-dlp`)

Where no visible transcript panel exists but the video carries closed captions, `yt-dlp` can pull the caption track directly:

```bash
yt-dlp --skip-download --write-subs --sub-langs en --convert-subs vtt "<video_url>"
```

This is used two ways:
- **Standalone**, when captions are the only thing available.
- **As a timing source**, specifically for PBS: the raw transcript (speaker labels, no timing) is scraped from the page per above, and the caption `.vtt` file is pulled separately purely to recover timestamps. A Python stitching script (`stitch_transcripts`, in `TranscriptExtraction.ipynb`) then aligns the two: it parses the VTT into timestamped caption blocks, and for each transcript line, searches nearby caption blocks for matching words to assign the closest timestamp, falling back to positional sequence if no word match is found.

**Current reliability note:** `yt-dlp` extraction works reliably on some platforms but has become unreliable on YouTube specifically, following recent site-side changes aimed at blocking automated downloading. Whether `yt-dlp` will work for a given source is effectively case-by-case — worth trying first, but not something to depend on for YouTube-hosted content. When it fails, path 3 (below) is the fallback.

### 3. Audio recording + WhisperX transcription

When neither a native transcript nor extractable captions are available (or `yt-dlp` is blocked), the documentary's audio is recorded directly and processed with [WhisperX](https://github.com/m-bain/whisperX), which extends OpenAI's Whisper with word-level timestamp alignment and speaker diarization. Pipeline steps (see `WhisperX_1_.ipynb`):

1. **Transcription** — `whisperx.load_model("large-v3", ...)` transcribes the audio in batches (`model.transcribe(audio, batch_size=...)`).
2. **Word-level alignment** — Whisper's native timestamps are segment-level and imprecise; `whisperx.load_align_model` + `whisperx.align(...)` refine these to word-level timing using a language-specific alignment model.
3. **Speaker diarization** — a separate Pyannote-based diarization pipeline (`whisperx.diarize.DiarizationPipeline`, requires a Hugging Face access token) identifies distinct speakers across the audio; `whisperx.assign_word_speakers(...)` then maps each transcribed word/segment to a speaker label.
4. **Formatting** — segments are converted into the standard `[MM:SS] SPEAKER: text` line format and written to a `.txt` file.

The notebook includes both a step-by-step version (useful for debugging one file at a time) and a batch version that pre-loads the Whisper and diarization models once, then loops over every supported audio file (`.mp3`, `.wav`, `.m4a`, `.flac`) in a directory — avoiding the cost of reloading multi-gigabyte models per file.

**Practical note:** because platform blocking has made `yt-dlp`-based extraction less reliable (particularly on YouTube), recording audio directly and relying on WhisperX has in practice become the more consistently workable path for many sources, even though it's more manual up front (requires playing/recording the full documentary) and computationally heavier (requires a GPU for practical runtime).

## Known limitations

- Speaker diarization (step 3 above) is probabilistic — it estimates *how many* distinct speakers there are and *which* speaker said what, and can make mistakes, especially with overlapping speech, short interjections, or similar-sounding voices. Diarization output should be treated as a strong starting point, not ground truth, and spot-checked.
- The caption-to-transcript stitching (PBS path) matches on shared words within a local window rather than exact alignment, so timestamps for stitched transcripts are approximate, not frame-accurate.
- DOM-scraping scripts are tied to each platform's current page structure and will break silently (return empty/malformed output) if the platform changes its markup.
