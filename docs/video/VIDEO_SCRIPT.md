# MedAnon: two-minute video script

Narration, natural English, about 275 words (110–115 s at a calm pace). Everything shown on screen is fictional
data, code, diagrams or aggregate metrics. Scene-by-scene timing and visuals: [`VIDEO_STORYBOARD.md`](VIDEO_STORYBOARD.md).

---

**[0:00 Problem]**
Hospitals want to use AI on doctor–patient conversations, but those conversations are full of private details:
names, ages, towns, employers. Bosch asked us to anonymize them without losing the medicine, on-premise, on a
small GPU.

**[0:13 Why it is hard]**
Two things make this hard. The dialogue is lowercase speech-to-text, so normal name detectors miss names. And
medicine is full of words that look like people: Cushing's, Parkinson's, drug brands. Mask those, and the note
loses its meaning.

**[0:28 Key idea]**
MedAnon never rewrites the text with a language model. Detectors propose small spans: rules for speech, spaCy, a
lowercase BERT and the OpenMed PII model. A clinical guard vetoes medical look-alikes. A policy decides what each
remaining span becomes, and every decision is logged.

**[0:47 Demo]**
Here is a fictional patient called Ray Cushing, who has Cushing's syndrome. His name becomes a patient placeholder.
The disease stays. His age becomes a decade band, his employer and city become placeholders, and his A1c becomes a
clinical range. The blood pressure stays exact.

**[1:07 Results]**
On fictional test worlds, the hardened release keeps every blood-pressure reading and hides exact counts of three or more children. On
our red-team probes it went from seven to twelve out of twelve. A linkage attacker's success dropped from about
twenty-two to twelve percent. On the evaluation set, the ICD-coding proxy didn't move at all.

**[1:30 Honesty and engineering]**
These are our own tests, not Bosch scores. Some risk remains: the clinical history itself can identify people.
Every decision is explained, it runs offline in under three gigabytes of GPU memory, and it passes 194 tests.

**[1:45 Close]**
MedAnon: remove the person, keep the patient.

---

Word count of the narration: about 275. If the recording runs long, cut the second sentence of "Results" first.
