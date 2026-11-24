"""Error analysis: Why GPT-5.4 collapses on Linguistic Feature (alpha=0.046 vs Flash 0.536)."""
import json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

gpt = json.load(open(ROOT / "Data/results/neurips_experiments/gpt54_P0_baseline/raw_scores.json"))
flash = json.load(open(ROOT / "Data/results/neurips_experiments/gemini_flash_P0_baseline/raw_scores.json"))
human = pd.read_csv(ROOT / "Data/processed/full_scores_both_annotators.csv").head(100)

dim = "Linguistic_Feature"
examples = []
gpt_all, flash_all, human_all = [], [], []

for i, row in human.iterrows():
    uuid = row["question_uuid"]
    hk = row["Linguistic_Feature_KC"]
    hl = row["Linguistic_Feature_LH"]
    h_cons = round((hk + hl) / 2)

    gpt_vals = gpt.get(uuid, {}).get(dim, [None, None, None])
    flash_vals = flash.get(uuid, {}).get(dim, [None, None, None])

    gv = [v for v in gpt_vals if v is not None]
    fv = [v for v in flash_vals if v is not None]

    gpt_med = int(np.median(gv)) if gv else None
    flash_med = int(np.median(fv)) if fv else None

    if gpt_med is not None:
        gpt_all.append(gpt_med)
    if flash_med is not None:
        flash_all.append(flash_med)
    human_all.append(h_cons)

    if gpt_med is not None and flash_med is not None:
        gpt_err = abs(gpt_med - h_cons)
        flash_err = abs(flash_med - h_cons)
        if gpt_err >= 2 and flash_err <= 1:
            examples.append({
                "idx": i, "uuid": uuid[:12],
                "human_K": int(hk), "human_L": int(hl), "consensus": h_cons,
                "gpt_runs": gpt_vals, "gpt_med": gpt_med,
                "flash_runs": flash_vals, "flash_med": flash_med,
            })

print(f"Found {len(examples)} items where GPT errs by 2+ but Flash within 1\n")
for e in examples[:5]:
    print(f"Item {e['idx']} ({e['uuid']}):")
    print(f"  Human: K={e['human_K']} L={e['human_L']} consensus={e['consensus']}")
    print(f"  GPT-5.4:  runs={e['gpt_runs']}  median={e['gpt_med']}  err={abs(e['gpt_med']-e['consensus'])}")
    print(f"  Flash:    runs={e['flash_runs']}  median={e['flash_med']}  err={abs(e['flash_med']-e['consensus'])}")
    print()

# Score distributions
print("Score distributions (Linguistic Feature):")
for label, arr in [("Human", human_all), ("GPT-5.4", gpt_all), ("Flash", flash_all)]:
    u, c = np.unique(arr, return_counts=True)
    dist = dict(zip(u.astype(int), c.astype(int)))
    total = sum(c)
    print(f"  {label:10s}: {dist}  mean={np.mean(arr):.2f}")

# Central tendency analysis
print(f"\nGPT mode=3 concentration: {sum(1 for x in gpt_all if x==3)}/{len(gpt_all)} = {sum(1 for x in gpt_all if x==3)/len(gpt_all)*100:.0f}%")
print(f"Flash mode=3 concentration: {sum(1 for x in flash_all if x==3)}/{len(flash_all)} = {sum(1 for x in flash_all if x==3)/len(flash_all)*100:.0f}%")
print(f"Human mode=2 concentration: {sum(1 for x in human_all if x==2)}/{len(human_all)} = {sum(1 for x in human_all if x==2)/len(human_all)*100:.0f}%")
