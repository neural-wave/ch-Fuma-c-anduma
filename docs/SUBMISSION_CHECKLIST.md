# Submission checklist: actions for the team (deadline Sunday 18 October 2026, 11:00 Europe/Zurich)

Done on branch `adversary-reidentification`: jury report (`report/JURY_REPORT.pdf`, 3 pages, LaTeX source and Markdown
mirror), README, MIT `LICENSE`, `THIRD_PARTY_NOTICES.md`, video script and storyboard, privacy clean-up of the attacker docs. The
items below need a person.

## Before 11:00

1. **Make the deliverables visible on the default branch.** `main` is still at the old build `ceff209` (old README,
   no report), and GitHub shows `main` by default. With the team's agreement, merge `adversary-reidentification`
   into `main` (or change the default branch). Check on github.com that the README and `report/JURY_REPORT.pdf` are
   what a visitor sees.
2. **Update the official repo `neural-wave/ch-Fuma-c-anduma`, which the jury report links to.** It will be
   **open-sourced after the event**. Push only the clean, history-free snapshot from the Windows machine, updated with
   these documents (`README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `report/`, video files) and with the evidence the
   report cites (`reports/FINAL_HARDENING.md`, `research/reid/`, `attack/paper/`). Before including `attack/`, remove
   or scrub the files listed under "Before any public release". Never push this repository's history there.
3. **Re-run the privacy gate on the Windows machine** with both real datasets, train and eval:
   `python scripts/privacy_check_docs.py` must report `FAIL 0`. On this Mac only the eval set was available: the gate
   reports `FAIL 0` there, and the training-set seeds could not be checked.
4. **Record and upload the video** (≤ 2:00) following `VIDEO_STORYBOARD.md` and `VIDEO_SCRIPT.md`. Close any
   `data/` or `outputs/` tabs first. Upload to SwissTransfer, then paste the link here and in the README.
   SwissTransfer link: `TODO`
5. **Deliver the anonymized eval JSONL: required by the task PDF ("Expected deliverables", item i).** Send the
   release output from the Windows machine (`outputs/final_release/…anonymized.jsonl`, SHA-256 `248a3d0f…`, 67/67 validated) through the
   channel the organizers name (ask them: repository, SwissTransfer or email). Do not send the older file in
   `outputs/submission/`. It is not in this repository on purpose, because the repository will be open-sourced.

## Decisions recorded (change them if the team disagrees)

- Team section: full names and emails, no roles (team decision; the guidelines ask for names and "possibly" roles).
- Licence: MIT (`LICENSE`). Third-party files keep their own licences (`THIRD_PARTY_NOTICES.md`); `common_words.tsv`
  stays under CC BY-SA 4.0.

## Before any public release

- Do not make this repository public as-is. Its history contains raw-data-derived material and old documents.
  Publish a clean snapshot without history and without data.
- The attacker's code, tests and JSON results still embed real encounter IDs and short record-specific phrases. The
  privacy gate only warns on `.py` files and does not scan `.json` or `.csv`. Affected: `attack/paper/attribute_review.py`,
  `attack/paper/mask_review.py`, `attack/paper/*.json`, `attack/paper/results/*.json`, `attack/handoff/*.json`,
  `attack/handoff/results/*.json`, `attack/tests/`. Their owner should scrub or exclude them from any public snapshot.
  This hand-off did not change attacker code, by design. `attack/paper/results/per_record.csv` is no longer tracked.
- Optional (guidelines step 5): prepare the 5-minute presentation; the report's diagram and tables can be reused.
