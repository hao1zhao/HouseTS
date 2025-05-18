# import timesfm
import torch
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
from baselines.finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from datasets import TimeSeriesDataset

from baselines.timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from baselines.timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
import numpy as np
from os import path
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from utils import compute_metrics

def get_model(context_len, horizon_len, load_weights: bool = False, checkpoint_path: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "google/timesfm-2.0-500m-pytorch"
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=128,
        horizon_len=horizon_len,
        num_layers=50,
        # use_positional_embedding=False,
        context_len=context_len,  # Context length can be anything up to 2048 in multiples of 32
    )
    tfm = TimesFm(hparams=hparams,
                  checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

    model = PatchedTimeSeriesDecoder(tfm._model_config)
    if load_weights:
        checkpoint_path = checkpoint_path if checkpoint_path else path.join(snapshot_download(repo_id), "torch_model.ckpt")
        loaded_checkpoint = torch.load(checkpoint_path)  # , weights_only=True)
        model.load_state_dict(loaded_checkpoint)
    return model, hparams, tfm#._model_config


def evaluation(
        model: TimesFm,
        val_dataset: Dataset,
        batch_size: int,
        save_path: 'results.npz'
) -> None:
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    labels, preds = [], []
    for batch in tqdm(val_loader, total=len(val_loader)):
        x_context, x_padding, freq, x_future = batch

        device = next(model.parameters()).device
        x_context = x_context.to(device)
        x_padding = x_padding.to(device)
        freq = freq.to(device)
        x_future = x_future.to(device)

        context_vals = x_context.unsqueeze(1).cpu().numpy()
        future_vals = x_future.unsqueeze(1).cpu().numpy()
        context_len = context_vals.shape[-1]
        horizon_len = future_vals.shape[-1]
        with torch.no_grad():
            predictions = model(x_context, x_padding.float(), freq)
            predictions_mean = predictions[..., 0]  # [B, N, 128]
            pred = predictions_mean[:, -1, context_len:context_len+horizon_len]  # [B, horizon_len]
        pred_vals = pred.unsqueeze(1).cpu().numpy()


        labels.append(future_vals)
        preds.append(pred_vals)
    labels, preds = np.concatenate(labels), np.concatenate(preds)

    np.savez(save_path, labels=labels, preds=preds, mean=val_dataset.scaler_target.mean_, std=val_dataset.scaler_target.scale_)
    metrics = compute_metrics(labels, preds, scaler=val_dataset.scaler_target)
    res = {'MSE': metrics[0], 'MAE': metrics[1], 'MAPE': metrics[2],
           'SMAPE': metrics[3], 'log-RMSE': metrics[4], 'r2': metrics[5]}
    print(res)


def single_gpu_example(model, tfm, train_dataset, val_dataset, test_dataset, batch_size, cp_path, save_path):
    """Basic example of finetuning TimesFM on stock data."""
    config = FinetuningConfig(batch_size=batch_size,
                              num_epochs=10,
                              learning_rate=1e-4,
                              use_wandb=True,
                              freq_type=1,
                              log_every_n_steps=5,
                              val_check_interval=0.5,
                              use_quantile_loss=True,
                              checkpoint_save_path=cp_path)

    finetuner = TimesFMFinetuner(model, tfm, config)

    print("\nStarting finetuning...")
    results = finetuner.finetune(train_dataset=train_dataset, val_dataset=val_dataset)

    print("\nFinetuning completed!")
    print(f"Training history: {len(results['history']['train_loss'])} epochs")

    evaluation(
        model=model,
        val_dataset=test_dataset,
        batch_size=batch_size,
        save_path=save_path,
    )


freq_type = 1
batch_size = 128
root_path = './data'
scaler = StandardScaler()
scaler.mean_, scaler.scale_ = [12.41033413], [0.72660783]
settings = [(6, 3), (6, 6)]
for context_len, horizon_len in settings:
    data_path = f'univariate_c{context_len}h{horizon_len}'
    train_dataset = TimeSeriesDataset(
        scaler,
        series=torch.load(path.join(root_path, data_path+'_train.pt')),
    )

    val_dataset = TimeSeriesDataset(
        scaler,
        series=torch.load(path.join(root_path, data_path+'_val.pt')),
    )

    test_dataset = TimeSeriesDataset(
        scaler,
        series=torch.load(path.join(root_path, data_path+'_test.pt')),
    )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Testing samples: {len(test_dataset)}")

    model, hparams, tfm = get_model(context_len, horizon_len, load_weights=True, checkpoint_path=f'./checkpoints/timesFM_univariate_c{context_len}h{horizon_len}.pt')

    print(f"Model loaded, evaluating model with zero-shot for c{context_len}h{horizon_len}...")
    evaluation(model, test_dataset, batch_size,
               f'./results/timesFM_univariate_zeroshot_c{context_len}h{horizon_len}.npz')
