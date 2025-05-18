import os, re, json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")          
DATA_ROOT = Path("")                   

def predict_next_price(history: dict[int, float]) -> float:
    """Send {year: price} history to GPT; return numeric prediction."""
    hist_txt = "\n".join(f"{y}: ${history[y]:,.0f}" for y in sorted(history))
    rsp = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.15,
        max_tokens=10,
        messages=[
            {"role": "system",
             "content": ("You are an experienced real-estate analyst. "
                         "Return ONLY the numeric prediction (no $ or commas).")},
            {"role": "user",
             "content": (f"Here are the year-end prices:\n\n{hist_txt}\n\n"
                         f"What is your prediction for {max(history)+1}?")},
        ],
    )
    txt = rsp.choices[0].message.content.strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", txt.replace(",", ""))
    if not m:
        raise ValueError(f"No number in reply: {txt}")
    return float(m.group())

# -------- Build workload --------
tasks = []
for f in DATA_ROOT.rglob("*_prices.json"):
    zip_code = f.parent.name
    with open(f) as fp:
        prices = {int(k): v for k, v in json.load(fp).items()}
    if len(prices) < 2:
        continue
    yrs = sorted(prices)
    tasks.append((zip_code,
                  {y: prices[y] for y in yrs[:-1]},  
                  yrs[-1],                           
                  prices[yrs[-1]]))                  

print("ZIPs to process:", len(tasks))

# -------- Batch inference --------
records = []
for z, hist, test_year, true_price in tqdm(tasks):
    try:
        pred = predict_next_price(hist)
        records.append({
            "zip": z,
            "test_year": test_year,
            "true_price": true_price,
            "pred_price": pred,
            "abs_err": abs(pred - true_price),
        })
    except Exception as e:
        print("Error", z, e)

df = pd.DataFrame(records)
if df.empty:
    print("No predictions produced.")
else:
    log_rmse = np.sqrt(((np.log(df.pred_price + 1) -
                         np.log(df.true_price + 1)) ** 2).mean())
    mape = (df.abs_err / df.true_price).mean() * 100
    print(f"Samples : {len(df)}")
    print(f"Log-RMSE: {log_rmse:.4f}")
    print(f"MAPE    : {mape:.2f}%")
    df.to_csv("gpt_price_preds.csv", index=False)
    print("Saved results to gpt_price_preds.csv")
