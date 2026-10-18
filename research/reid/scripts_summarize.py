"""Print the report tables from a results JSON (dev pooled or a single world)."""
import json
import sys

d = json.load(open(sys.argv[1]))
src = d["pooled"] if "pooled" in d else None
POLS = ["original", "default", "age_bands", "age_bands_strict_dates"]


def f(x):
    if isinstance(x, list) and x[0] is None:
        return "n/a"
    return "-" if x is None else f"{x[0]*100:.1f} ({x[1]*100:.1f}-{x[2]*100:.1f})" if isinstance(x, list) else f"{x*100:.1f}"


def get(p, t):
    if src:
        return src.get(p, {}).get(t)
    w = next(iter(d["worlds"]))
    r = d["worlds"][w].get(p, {}).get(t)
    return r["metrics"] if r else None


print("| Policy | Thr | FS top-1 | FS top-5 | FS success (closed) | FS precision | FS success (open) | FS false match, absent | Det. success | Det. precision | Patients re-id |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p in POLS:
    for t in "ABC":
        r = get(p, t)
        if not r:
            continue
        c, o = r["closed/2000"], r["open/2000"]
        print(f"| {p} | {t} | {f(c['fs'].get('top1'))} | {f(c['fs'].get('top5'))} | {f(c['fs']['success_over_eligible'])} | "
              f"{f(c['fs']['precision'])} | {f(o['fs']['success_over_eligible'])} | {f(o['fs'].get('false_match_rate_absent'))} | "
              f"{f(c['det']['success_over_eligible'])} | {f(c['det']['precision'])} | {f(c['fs'].get('patient_reidentified'))} |")
print("\nPool size (FS closed top-1 / success):")
for p in POLS:
    for t in "ABC":
        r = get(p, t)
        if r:
            print(p, t, " ".join(f"P={P}: {r[f'closed/{P}']['fs']['top1'][0]*100:.1f}/{r[f'closed/{P}']['fs']['success_over_eligible'][0]*100:.1f}"
                                 for P in (100, 500, 2000, 10000)), "| hard:", r["closed/2000"]["fs"].get("hard_success"),
                  "hard FM absent:", r["open/2000"]["fs"].get("false_match_rate_absent_hard"))
if "paired" in d:
    print("\nPaired (FS / det success diff, CI):")
    for k, v in d["paired"].items():
        if "/2000" in k and "closed" in k:
            print(k, {m: (v[m]["diff"], v[m]["ci"], v[m]["discordant_a_only"], v[m]["discordant_b_only"]) for m in v})
for w, W in d["worlds"].items():
    for p in POLS:
        for t in "ABC":
            x = W.get(p, {}).get(t)
            if x and x.get("inference"):
                print("inference", w, p, t, x["inference"])
    if "controls" in W:
        print("controls", w, json.dumps({n: {p: {t: {k: (v["fs"].get("top1", [None])[0] if isinstance(v, dict) and "fs" in v else v,
                                                         v["fs"]["success_over_eligible"][0] if isinstance(v, dict) and "fs" in v else None)
                                                     for k, v in tv.items()} for t, tv in pv.items()} for p, pv in nv.items()}
                                          for n, nv in W["controls"].items()}))
    for p in POLS:
        st = W.get(p, {}).get("default", None)
    for p in POLS:
        sh = W.get(p, {}).get("C", {}).get("shuffled_truth")
        if sh:
            print("shuffled truth", w, p, "C", {k: (v["fs"].get("top1"), v["fs"]["success_over_eligible"]) for k, v in sh.items()})
