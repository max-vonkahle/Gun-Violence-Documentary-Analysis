# Project Overview (Plain-Language Guide)

## The question

Documentaries about gun violence often feature many kinds of voices: grieving family members, survivors, policy experts, journalists, activists, law enforcement. This project asks a simple question with a complicated answer: **if you strip away the video and just look at the words, do these documentaries share common themes — and does it matter *who* is speaking?**

For example, do bereaved family members talk about different things than policy experts, even within the same film? Do certain themes (grief, legislation, mental health, community trauma) show up across many documentaries regardless of who's speaking, or are some themes tightly tied to a specific kind of speaker?

Answering this by hand — reading transcripts of many documentaries and manually tagging themes — would take an enormous amount of time and would be shaped by whatever the reader happens to notice first. This project instead uses computational tools to do a first pass at this pattern-finding, so the human researcher can then review, question, and refine what the tools find.

## The process, step by step

### Step 1: Get a transcript

Before any analysis can happen, each documentary needs to become text: a transcript that records not just *what* was said, but *when* it was said and *who* said it.

There are three ways this project gets there, tried in order of preference:

1. **Take it from the source.** Some streaming platforms that host these documentaries (like Kanopy, Alexander Street, or PBS) already display a transcript alongside the video. When available, this is the fastest and most reliable path, and it's usually created by a human, not a machine.
2. **Pull the captions.** If there's no visible transcript but the video has closed captions, those can sometimes be extracted directly using a video-downloading tool. This gives timing information but not always clean speaker labels, so extra work goes into matching caption timing to the transcript text.
3. **Record and transcribe with AI.** If neither of the above works — for instance, some platforms have recently updated their systems specifically to block automated caption downloading — the fallback is to record the documentary's audio directly and run it through an AI speech-recognition tool called **WhisperX**. WhisperX doesn't just transcribe speech to text; it also figures out *when* each word was spoken and uses a separate AI process to guess *who* is speaking at any given moment (a process called "speaker diarization").

Whichever path is used, the end result looks the same: a text file where each line has a timestamp, a speaker label, and what was said — for example, `[04:12] BEREAVED: I still don't understand why he was there.` Making all three paths produce the same format matters because it means everything downstream doesn't need to know or care how a particular transcript was obtained.

### Step 2: Group the speakers into categories

Rather than treating every individual speaker as unique, speakers get sorted into broader categories — for example, "bereaved family member," "policy expert," "journalist," "law enforcement." This makes it possible to ask questions like "do bereaved family members across different documentaries tend to talk about similar things?" instead of only being able to compare one specific person to another.

### Step 3: Let the computer find topics

This is where the AI topic-modeling comes in. The tool used, called **BERTopic**, works by:
- Converting chunks of text into a kind of mathematical fingerprint that captures meaning (not just keywords)
- Grouping chunks whose fingerprints are similar into clusters
- Labeling each cluster with the words that best represent it — these become the "topics"

The interesting part of this project is that there isn't just one way to do this. The research is testing **several different variations** of the process — different ways of chunking the transcript (by time, by speaker turn) and different ways of using the speaker-category information. Comparing these approaches is itself part of the research question: which method actually produces the most trustworthy, meaningful topics? (See `methodology.md` for the technical comparison of the approaches implemented so far.)

### Step 4: Interpret the results

None of this is meant to replace human judgment. The output of the topic modeling is a starting point — a way to surface patterns across dozens of hours of documentary footage that would be very hard to notice by reading alone. A researcher still needs to look at what the model finds, decide whether it makes sense, and dig into the actual transcript passages behind each topic.

The visualizations below show the current results — each corresponds to a different chunking approach and can be explored interactively:

- [Time-Window Approach](https://max-vonkahle.github.io/Gun-Violence-Documentary-Analysis/visualizations/Time_Window_Bertopic.html) — transcripts divided into fixed 60-second windows regardless of who is speaking
- [Speaker-Turn Approach](https://max-vonkahle.github.io/Gun-Violence-Documentary-Analysis/visualizations/Turn_Script_Bertopic.html) — transcripts divided along speaker-turn boundaries

## Why this matters

Gun violence documentaries are made by different filmmakers, with different focuses, often years apart. If common themes appear across many of them regardless of speaker, that says something about how our culture consistently frames this issue. If themes split sharply by *who* is speaking, that says something about whose perspectives get represented, and how — which is itself worth studying as a piece of media analysis, independent of the technical experiment.

---

*If any of these terms or steps are unclear, `data-acquisition.md` and `methodology.md` cover the technical side in more depth — but this document should stand alone for understanding what the project is doing and why.*
