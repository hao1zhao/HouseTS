import os, re, json, time, base64, mimetypes
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")           

DATA_ROOT  = Path("")                  
MODEL      = "gpt-4o"                            
TEMP       = 0.15                                     
COOLDOWN_EACH, SECS = 25, 60                           
YEAR_RE = re.compile(r"^(\d{4})")                      

# ---------- helpers ----------
def img_to_b64(p: Path) -> str:
    mime, _ = mimetypes.guess_type(p)
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()

def parse_price(text: str) -> float:
    
    try:
        return float(json.loads(text)["price"])
    except Exception:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        if not m:
            raise ValueError(f"No number in: {text[:60]} …")
        return float(m.group())

def gpt_predict(hist: dict[int, float],
                semantics: dict,
                images_b64: list[str]) -> float:
    years   = sorted(hist)
    prices  = "\n".join(f"{y}: ${hist[y]:,.0f}" for y in years)
    system  = ("You are a real-estate analyst combining satellite images, "
               "semantic descriptions, and historical prices. "
               'Return ONLY {"price": <number>} (no $ or commas).')
    user = [
        *[{"type": "image_url", "image_url": {"url": b64}} for b64 in images_b64],
        {"type": "text",
         "text": (
             f"Satellite semantics:\n"
             f"- Summary: {semantics['trend_summary']}\n"
             f"- Keywords: {', '.join(semantics['keywords'])}\n"
             f"- Notable changes: {'; '.join(semantics['notable_changes'][:3])}\n\n"
             f"Year-end prices:\n{prices}\n\n"
             f"What is your prediction for {years[-1]+1}?"
         )},
    ]
    rsp = openai.chat.completions.create(
        model=MODEL,
        temperature=TEMP,
        max_tokens=12,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
    )
    return parse_price(rsp.choices[0].message.content.strip())

records, errors = [], []
zip_folders = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
print("ZIP folders:", len(zip_folders))

for i, zdir in enumerate(tqdm(zip_folders, desc="Predicting")):
    try:
        p_json, a_json = zdir / "prices.json", zdir / "analysis.json"
        if not p_json.exists() or not a_json.exists():
            continue

        price_map = {int(k): v for k, v in json.loads(p_json.read_text()).items()}
        years_sorted = sorted(price_map)
        if len(years_sorted) < 2:
            continue

        hist        = {y: price_map[y] for y in years_sorted[:-1]}
        test_year   = years_sorted[-1]
        true_price  = price_map[test_year]

        
        imgs = []
        for p in zdir.glob("*.png"):
            m = YEAR_RE.match(p.stem)
            if not m:
                continue
            year = int(m.group(1))
            if year in hist:
                imgs.append((year, p))
        if not imgs:
            continue

        imgs.sort(key=lambda t: t[0])
        imgs_b64 = [img_to_b64(p) for _, p in imgs]
        semantics = json.loads(a_json.read_text())

        pred_price = gpt_predict(hist, semantics, imgs_b64)
        records.append({"zip": zdir.name,
                        "test_year": test_year,
                        "true": true_price,
                        "pred": pred_price})
    except Exception as e:
        errors.append((zdir.name, str(e)))

    if (i + 1) % COOLDOWN_EACH == 0:
        print(f"Cooling {SECS}s …")
        time.sleep(SECS)

df = pd.DataFrame(records)
if df.empty:
    print("No successful predictions.")
else:
    df.to_csv("gpt_allmodal_predictions.csv", index=False)
    eps = 1.0
    log_rmse = np.sqrt(((np.log(df.pred + eps) - np.log(df.true + eps))**2).mean())
    mape = ((df.pred - df.true).abs() / df.true).mean() * 100
    print(f"\nZIPs processed : {len(df)}")
    print(f"Log-RMSE        : {log_rmse:.4f}")
    print(f"MAPE            : {mape:.2f}%")

if errors:
    print(f"\n{len(errors)} ZIPs failed. Example:")
    for z, msg in errors[:5]:
        print(f"  {z}: {msg}")
