# MedAnon: video storyboard (≤ 2:00)

Narration text: [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md). Target length 1:50. Show only fictional encounters, code,
diagrams and aggregate metrics. Never show `data/` or `outputs/`: close those editor tabs and collapse the folders
before recording.

| # | Time | On screen | Narration (short) |
|---|---|---|---|
| 1 | 0:00–0:13 | Title card "MedAnon: anonymization of doctor–patient encounters", Bosch challenge, Neural Wave 2026. Then the fictional demo dialogue (`medanon demo`) with the private details highlighted: *ray cushing*, *46-year-old*, *cashier at boeing*, *boise*. Three chips: "on-premise", "≤ 4 GB VRAM", "privacy · utility · originality and scalability, equal weight". | Hospitals want to use AI on doctor–patient conversations… |
| 2 | 0:13–0:28 | Split screen: a lowercase dialogue line next to the cased note. Highlight "cushing" twice: once as a surname, once in "cushing's syndrome". Caption: "same word, two meanings". | The dialogue is lowercase… medicine is full of words that look like people… |
| 3 | 0:28–0:47 | Pipeline diagram (README / report §2), boxes lighting up left to right: detectors → filters + clinical veto → person registry → overlaps → policy → dataset step → JSONL + audit + manifest. Small label: "no generative model at inference". | Detectors propose small spans… a clinical guard vetoes… every decision is logged. |
| 4 | 0:47–1:07 | Terminal recording of the demo, before/after side by side. Zoom on: `[PATIENT_NAME]` · "cushing's syndrome" kept · `[40-49]` · `[EMPLOYER]` · `[CITY]` · `[8-9]` · blood pressure line unchanged. Then one explanation line from the demo output. | Here is a fictional patient called Ray Cushing… |
| 5 | 1:07–1:30 | Four result cards, each with its source label. BP kept exactly 44/92 → 92/92 (Synthea dev1). Red-team probes 7/12 → 12/12. Linkage success 21.5 % → 11.7 % (synthetic dev worlds, not blind). ICD-10 proxy F1 Δ 0.0 (local proxy, 67 eval records). Footer: "Local synthetic tests and proxies, not Bosch scores." | On fictional test worlds… |
| 6 | 1:30–1:45 | Two columns. Left, "Residual risk": clinical history can link · ambiguous surnames. Right, "Engineering": every decision explained (989/989 on the eval output) · offline · 2,639 MiB peak VRAM · 194 tests. | These are our own tests… |
| 7 | 1:45–1:50 | Closing card: "MedAnon: remove the person, keep the patient." Team names, repository URL, "Report: report/JURY_REPORT.pdf". | MedAnon: remove the person, keep the patient. |

## Recording the demo (scene 4)

On a machine with prepared models (`medanon prepare-models` done once), from the repository root:

```bash
medanon demo --offline                    # release profile, fictional encounter, before / after / explanations
```

Without model weights (for example a laptop without the transformer models), use the reduced profile and say so on
screen ("reduced profile: rules + spaCy"):

```bash
medanon demo --no-ml --offline
```

The before/after lines in scene 4 were checked with the reduced profile. Record the profile you actually run. The
release profile can differ on individual spans; do not edit the output.

Optional second clip, also fictional: the 12 red-team probes. Write the output to a scratch path outside the
repository:

```bash
medanon anonymize --offline -i tests/fixtures/redteam_handoff/encounters.jsonl -o /tmp/redteam.anon.jsonl
medanon validate --original tests/fixtures/redteam_handoff/encounters.jsonl --anonymized /tmp/redteam.anon.jsonl
```

## Export and delivery

- MP4, 1080p, at most 2:00. Burn in subtitles if the narration is quiet.
- Upload to SwissTransfer and paste the link into `SUBMISSION_CHECKLIST.md` and the README.
- Visuals for scenes 3 and 5 can be taken from `report/JURY_REPORT.pdf` (diagram, tables).
