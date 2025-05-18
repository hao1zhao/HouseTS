import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from baselines.chronos import ChronosPipeline
from datasets import TimeSeriesDataset
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from utils import compute_metrics
from os import path

def evaluation(
        pipeline: ChronosPipeline,
        val_dataset: Dataset,
        context_len: int,
        horizon_len: int,
        batch_size: int,
        save_path: Optional[str] = "predictions.npy",
) -> None:
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    labels, preds = [], []
    for batch in tqdm(val_loader, total=len(val_loader)):
        x_context, x_padding, freq, x_future = batch
        with torch.no_grad():
            predictions = pipeline.predict(x_context, horizon_len) 
        pred_vals = predictions.mean(dim=1, keepdim=True).cpu().numpy()
        future_vals = x_future.unsqueeze(1).numpy()


        labels.append(future_vals)
        preds.append(pred_vals)
    labels, preds = np.concatenate(labels), np.concatenate(preds)

    np.savez(save_path, labels=labels, preds=preds, mean=val_dataset.scaler_target.mean_,
             std=val_dataset.scaler_target.scale_)
    metrics = compute_metrics(labels, preds, scaler=val_dataset.scaler_target)
    res = {'MSE': metrics[0], 'MAE': metrics[1], 'MAPE': metrics[2],
           'SMAPE': metrics[3], 'log-RMSE': metrics[4], 'r2': metrics[5]}
    print(res)


batch_size = 128
root_path = './data/HouseTS'
scaler = StandardScaler()
scaler.mean_, scaler.scale_ = [12.41033413], [0.72660783]
#
settings = [(6, 3), (6, 6), (6, 12), (12, 3), (12, 6), (12, 12)]
for i, (context_len, horizon_len) in enumerate(settings):
    data_path = f'univariate_c{context_len}h{horizon_len}'
    test_dataset = TimeSeriesDataset(
        scaler,
        series=torch.load(path.join(root_path, data_path+'_test.pt')),
    )

    print(f"Created datasets:")
    print(f"- Testing samples: {len(test_dataset)}")
    print(i, context_len, horizon_len, '----------------------')

    pipeline = ChronosPipeline.from_pretrained(
      "amazon/chronos-t5-base",
      device_map=device,
      torch_dtype=torch.bfloat16,
    ) 
    print('zeroshot...')
    evaluation(pipeline, test_dataset, context_len, horizon_len, batch_size, f'./results/Chronos_univariate_zeroshot_c{context_len}h{horizon_len}.npz')

    print('finetune...')
    pipeline = ChronosPipeline.from_pretrained(
      f'checkpoints/run-{i}/checkpoint-final',#"amazon/chronos-t5-base",
      device_map=device,
      torch_dtype=torch.bfloat16,
    ) 

    evaluation(pipeline, test_dataset, context_len, horizon_len, batch_size, f'./results/Chronos_univariate_finetune_c{context_len}h{horizon_len}.npz')
   