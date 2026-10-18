"""Build the compact lexicons shipped in ``src/medanon/resources/`` from public-domain sources.

Run once (requires internet + ``pip install wordfreq simple-icd-10-cm``)::

    python scripts/build_resources.py

Sources (all US-government public-domain data):
  * US Census 2010 surnames            -> src/medanon/resources/surnames.tsv
  * US Census 1990 first names + SSA top-1000 baby names (hadley/data-baby-names mirror)
                                       -> src/medanon/resources/first_names.tsv
  * FDA NDC product directory          -> src/medanon/resources/drugs.txt
  * ICD-10-CM (via simple-icd-10-cm)   -> src/medanon/resources/icd_vocab.tsv, resources/eponyms.tsv
  * wordfreq (English frequencies)     -> src/medanon/resources/common_words.tsv

The inference pipeline only reads the generated files; it never touches the network.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "src" / "medanon" / "resources"
CACHE = ROOT / ".cache" / "downloads"

URLS = {
    "surnames": "https://www2.census.gov/topics/genealogy/2010surnames/names.zip",
    "female_1990": "https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first",
    "male_1990": "https://www2.census.gov/topics/genealogy/1990surnames/dist.male.first",
    "babynames": "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv",
    "ndc": "https://www.accessdata.fda.gov/cder/ndctext.zip",
}
ALPHA = re.compile(r"^[a-z][a-z'\-]*[a-z]$")


def fetch(key: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / key
    if not path.exists():
        print(f"downloading {URLS[key]}")
        req = urllib.request.Request(URLS[key], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def zipf(word: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(word, "en")


def build_names() -> None:
    first: dict[str, float] = defaultdict(float)  # name -> max popularity (percent of births/population)
    for key in ("female_1990", "male_1990"):
        for line in fetch(key).decode("latin-1").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                first[parts[0].lower()] = max(first[parts[0].lower()], float(parts[1]))
    reader = csv.DictReader(io.StringIO(fetch("babynames").decode("utf-8")))
    for row in reader:
        name = row["name"].lower()
        first[name] = max(first[name], float(row["percent"]) * 100)
    with open(RES / "first_names.tsv", "w", encoding="utf-8", newline="\n") as f:
        for name, pop in sorted(first.items()):
            if ALPHA.match(name) and len(name) >= 2:
                f.write(f"{name}\t{pop:.5f}\n")
    print("first names:", len(first))

    zf = zipfile.ZipFile(io.BytesIO(fetch("surnames")))
    rows = csv.DictReader(io.TextIOWrapper(zf.open("Names_2010Census.csv"), encoding="latin-1"))
    n = 0
    with open(RES / "surnames.tsv", "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            name, count = row["name"].lower(), row["count"]
            if not count.isdigit() or int(count) < 300 or not ALPHA.match(name) or len(name) < 2:
                continue
            f.write(f"{name}\t{count}\n")
            n += 1
    print("surnames:", n)


def build_drugs() -> None:
    zf = zipfile.ZipFile(io.BytesIO(fetch("ndc")))
    text = zf.open("product.txt").read().decode("latin-1")
    rows = csv.DictReader(io.StringIO(text), delimiter="\t")
    reliable_brand_cats = {"NDA", "ANDA", "BLA", "NDA AUTHORIZED GENERIC"}
    product_types = {"HUMAN PRESCRIPTION DRUG", "HUMAN OTC DRUG", "VACCINE", "PLASMA DERIVATIVE"}
    terms: set[str] = set()
    rows = list(rows)
    # company / store-brand names (Kroger, Walgreens, Pfizer ...) are labelers, not drugs
    labelers = {tok for row in rows for tok in re.split(r"[^a-z\-]+", row["LABELERNAME"].lower()) if len(tok) >= 4}
    for row in rows:
        if row["PRODUCTTYPENAME"] not in product_types or "HOMEOPATHIC" in row["MARKETINGCATEGORYNAME"]:
            continue
        # allergen extracts are named after plants/places/people (Johnson grass, Timothy, Arizona cypress)
        if "Allergens" in (row["PHARM_CLASSES"] or "") or "ALLERGENIC" in row["PRODUCTTYPENAME"]:
            continue
        for tok in re.split(r"[^a-z\-]+", row["SUBSTANCENAME"].lower()):  # active ingredients only
            if len(tok) >= 4 and ALPHA.match(tok):
                terms.add(tok)
        if row["MARKETINGCATEGORYNAME"] in reliable_brand_cats:
            for tok in re.split(r"[^a-z\-]+", row["PROPRIETARYNAME"].lower()):
                # brand-name tokens: drop ordinary English words ("relief", "extra", "day", ...)
                if len(tok) >= 4 and ALPHA.match(tok) and zipf(tok) < 3.8 and tok not in labelers:
                    terms.add(tok)
    # a drug token must not double as a common word, a place or a reasonably common personal name;
    # the curated clinical_terms.txt still covers the well-known ambiguous brands (Allegra ...)
    first = {l.split("\t")[0]: float(l.split("\t")[1]) for l in open(RES / "first_names.tsv", encoding="utf-8")}
    sur = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(RES / "surnames.tsv", encoding="utf-8")}
    places = {l.split("\t")[0] for l in open(RES / "places.tsv", encoding="utf-8") if not l.startswith("#")}
    dropped = {t for t in terms if zipf(t) >= 4.3 or t in places or first.get(t, 0) >= 0.02 or sur.get(t, 0) >= 1000}
    terms -= dropped
    with open(RES / "drugs.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(terms)) + "\n")
    print("drug tokens:", len(terms), "dropped as ambiguous:", len(dropped))


def build_icd() -> None:
    import simple_icd_10_cm as cm

    vocab: Counter[str] = Counter()
    eponyms: dict[str, Counter[str]] = defaultdict(Counter)
    for code in cm.get_all_codes(with_dots=True):
        desc = cm.get_description(code)
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", desc)
        for tok in {t.lower().rstrip("'").removesuffix("'s") for t in toks}:
            if len(tok) >= 3:
                vocab[tok] += 1
        for i, tok in enumerate(toks):
            # ICD-10-CM descriptions are sentence case, so a capitalised non-initial word is an
            # eponym / proper noun used medically (Parkinson's disease, Bell's palsy, Lyme ...).
            possessive_first = i == 0 and re.fullmatch(r"[A-Z][a-z]+('s|s')", tok)
            if (i > 0 and re.fullmatch(r"[A-Z][a-z][a-z'\-]*", tok)) or possessive_first:
                head = toks[i + 1].lower() if i + 1 < len(toks) else ""
                eponyms[tok.lower().rstrip("'").removesuffix("'s")][head] += 1
    with open(RES / "icd_vocab.tsv", "w", encoding="utf-8", newline="\n") as f:
        for tok, c in sorted(vocab.items()):
            f.write(f"{tok}\t{c}\n")
    with open(RES / "eponyms.tsv", "w", encoding="utf-8", newline="\n") as f:
        for ep, heads in sorted(eponyms.items()):
            if len(ep) >= 3:
                top = ",".join(h for h, _ in heads.most_common(8) if h)
                f.write(f"{ep}\t{sum(heads.values())}\t{top}\n")
    print("icd vocab:", len(vocab), "eponyms:", len(eponyms))


def build_common_words() -> None:
    from wordfreq import top_n_list, zipf_frequency

    with open(RES / "common_words.tsv", "w", encoding="utf-8", newline="\n") as f:
        for w in top_n_list("en", 60000):
            if re.fullmatch(r"[a-z][a-z']*", w):
                f.write(f"{w}\t{zipf_frequency(w, 'en'):.2f}\n")
    print("common words written")


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    steps = sys.argv[1:] or ["names", "drugs", "icd", "common"]
    for step in steps:
        {"names": build_names, "drugs": build_drugs, "icd": build_icd, "common": build_common_words}[step]()
