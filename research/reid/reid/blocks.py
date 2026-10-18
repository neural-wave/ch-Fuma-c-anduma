"""QI-free clinical content blocks that give encounters realistic length and structure (history details, review of
systems, examination, plan, family history). Same parity split into dev/test families as render.py."""
from __future__ import annotations

SEVERITY = ["it's maybe a four out of ten", "honestly like a seven", "it comes and goes", "worse at night",
            "about a five , six on a bad day", "it's not terrible but it's annoying"]
TRIED = ["i took some tylenol and it helped a little", "i tried ibuprofen , didn't do much",
         "i've just been resting", "i used a heating pad", "nothing really , i figured it would go away",
         "my wife made me drink a lot of tea", "i took some cough syrup from the drugstore", "ice , mostly"]
ROS = [  # (doctor question, negative answer, positive answer, note item)
    ("any fevers or chills ?", "no fevers", "i felt a little feverish", "Constitutional: {} fever or chills."),
    ("any chest pain ?", "no , no chest pain", "a little tightness sometimes", "Cardiovascular: {} chest pain."),
    ("shortness of breath ?", "no", "only when i climb stairs", "Respiratory: {} shortness of breath."),
    ("any nausea or vomiting ?", "no", "some nausea yeah", "Gastrointestinal: {} nausea or vomiting."),
    ("how about headaches ?", "not really", "a few , yeah", "Neurological: {} headaches."),
    ("any trouble sleeping ?", "i sleep okay", "yeah , i wake up a lot", "Psychiatric: {} sleep disturbance."),
    ("any rashes or skin changes ?", "no", "a small rash on my arm", "Skin: {} rash."),
    ("any burning when you urinate ?", "no", "a little", "Genitourinary: {} dysuria."),
    ("changes in your appetite ?", "no , same as always", "i haven't been very hungry", "Constitutional: {} appetite change."),
    ("any dizziness ?", "no", "when i stand up too fast", "Neurological: {} dizziness."),
    ("any swelling in your legs ?", "no", "my ankles by the end of the day", "Cardiovascular: {} lower extremity edema."),
    ("how's your mood been ?", "pretty good", "kind of down lately", "Psychiatric: {} depressed mood."),
]
EXAM_D = ["i'm going to listen to your heart and lungs now .", "take a deep breath for me . and again .",
          "your lungs sound clear .", "your heart sounds regular , no murmurs .", "let me look in your ears and throat .",
          "your throat is a little red .", "i'm going to press on your belly , tell me if it hurts .",
          "no tenderness there , good .", "let me check your reflexes .", "squeeze my fingers . good .",
          "any pain when i press here ?", "i'm just going to feel your neck for lymph nodes .",
          "there's a little swelling here .", "follow my finger with your eyes ."]
EXAM_N = ["General: Alert, in no acute distress.", "HEENT: Normocephalic, atraumatic. Oropharynx clear.",
          "Neck: Supple, no lymphadenopathy.", "Cardiovascular: Regular rate and rhythm, no murmurs.",
          "Respiratory: Clear to auscultation bilaterally.", "Abdomen: Soft, nontender, nondistended.",
          "Extremities: No edema.", "Neurological: Alert and oriented, no focal deficits.",
          "Skin: Warm and dry, no rashes.", "Musculoskeletal: Full range of motion, no tenderness.",
          "Psychiatric: Normal mood and affect.", "Eyes: Pupils equal and reactive."]
PLAN_D = ["i want you to keep an eye on your symptoms and call us if anything gets worse .",
          "we'll get some blood work today just to be safe .", "make sure you're drinking plenty of fluids .",
          "i'm going to send a prescription to your pharmacy .", "let's keep the same doses for now .",
          "try to cut back on salt and get some walking in .", "if you get a fever over a hundred and one , call me .",
          "i'd like you to see a specialist about this .", "we'll recheck your labs before the next visit .",
          "i'll put in a referral for physical therapy ."]
PLAN_N = ["Continue current medications.", "Labs ordered: CBC, CMP.", "Patient educated on warning signs.",
          "Lifestyle modification counseled.", "Referral placed.", "Medication reconciliation completed.",
          "Return precautions reviewed.", "Encouraged hydration and rest."]
FAMILY = ["Mother with type 2 diabetes.", "Father had a myocardial infarction at age 62.", "Sister with asthma.",
          "Mother with breast cancer diagnosed at 58.", "No significant family history.", "Father with hypertension.",
          "Maternal grandmother with Alzheimer disease.", "Brother with hyperlipidemia."]
ALLERGY = ["No known drug allergies.", "Penicillin (rash).", "Sulfa drugs.", "NKDA.", "Codeine (nausea).",
           "Seasonal allergies only."]


def hpi_dialogue(r, rk: str, lay: str, child: bool) -> list[tuple[str, str]]:
    """(speaker, text) turns that expand the history for the visit reason."""
    t = []
    if rk == "acute":
        t += [("D", r.pick(["how bad is it ?", "on a scale of one to ten , how would you rate it ?", "is it getting worse ?",
                            "how is it affecting your day ?"])), ("P", r.pick(SEVERITY))]
        t += [("D", r.pick(["have you tried anything for it ?", "did you take anything at home ?", "anything help ?",
                            "what have you done so far ?"])), ("P", r.pick(TRIED))]
        if r.p(0.5):
            t += [("D", r.pick(["anyone sick at home ?", "any sick contacts ?"])),
                  ("P", r.pick(["my kid had something last week", "not that i know of", "a couple people at work", "no"]))]
    elif rk == "chronic":
        t += [("D", r.pick([f"how have you been managing the {lay} at home ?", "are you checking your numbers at home ?",
                            "any side effects from the medicine ?", "have you been taking everything as prescribed ?"])),
              ("P", r.pick(["mostly yes , i miss a dose here and there", "my readings have been around one thirty over eighty",
                            "no side effects that i noticed", "i've been better about it since the last visit"]))]
        t += [("D", r.pick(["how is your diet ?", "are you getting any exercise ?", "how's your energy ?",
                            "any new symptoms ?"])),
              ("P", r.pick(["could be better , i'm trying", "i walk most days", "pretty tired honestly",
                            "nothing new , i think"]))]
    else:
        t += [("D", r.pick(["any concerns you want to bring up today ?", "anything new since last year ?",
                            "how have you been feeling overall ?", "any questions for me before we start ?"])),
              ("P", r.pick(["not really , just tired sometimes", "no , i feel good", "i've had a few headaches",
                            "just want to make sure everything's okay"] if not child else
                           ["he's been growing a lot", "she's eating well", "no concerns , really",
                            "just the usual colds from daycare"]))]
    return t


def ros(r, k: int) -> tuple[list[tuple[str, str]], list[str]]:
    items = r.rng.sample(ROS[r.fam::2], min(k, len(ROS[r.fam::2])))
    dia, note = [], []
    for q, neg, pos, item in items:
        positive = r.p(0.2)
        dia += [("D", q), ("P", pos if positive else neg)]
        note.append(item.format("Reports" if positive else "Denies"))
    return dia, note


def exam(r, k_d: int, k_n: int) -> tuple[list[str], list[str]]:
    return r.rng.sample(EXAM_D[r.fam::2], min(k_d, 7)), r.rng.sample(EXAM_N[r.fam::2], min(k_n, 6))


def plan(r, k_d: int, k_n: int) -> tuple[list[str], list[str]]:
    return r.rng.sample(PLAN_D[r.fam::2], min(k_d, 5)), r.rng.sample(PLAN_N[r.fam::2], min(k_n, 4))
