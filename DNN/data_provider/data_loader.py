import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class Dataset_Custom(Dataset):
    _shared_scaler = None

    def __init__(self, args, root_path, data_path, flag='train', size=None, features='MS', target='price', freq='M', train_ratio=0.5, val_ratio=0.4, scale=True):
        super().__init__()
        if size is None:
            self.seq_len = 6
            self.label_len = 3
            self.pred_len = 6
        else:
            self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self.features = features.upper()
        self.target = target
        self.freq = freq
        self.scale = scale
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.root_path = root_path
        self.data_path = data_path
        self.series_list = []
        self.out_list = []
        self.xmark_list = []
        self.index_map = []
        self.__read_data__()

    def __read_data__(self):
        fpath = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(fpath, parse_dates=['date'])
        df_raw = df_raw.sort_values(['zipcode', 'date']).reset_index(drop=True)
        df_raw['year'] = df_raw['date'].dt.year
        df_raw['month'] = df_raw['date'].dt.month
        df_raw.drop(columns=['city', 'city_full'], inplace=True, errors='ignore')
        time_mark_cols = ['year', 'month']
        base_remove = ['date', 'zipcode', self.target] + time_mark_cols
        numeric_cols = [c for c in df_raw.columns if c not in base_remove]
        if self.features == 'S':
            enc_cols = [self.target]
        elif self.features in ('M', 'MS'):
            enc_cols = numeric_cols
        else:
            raise ValueError
        grouped = df_raw.groupby('zipcode')
        sub_dfs = []
        for zipcode, gdf in grouped:
            gdf = gdf.sort_values('date')
            N = len(gdf)
            if N < (self.seq_len + self.pred_len):
                continue
            train_cnt = int(N * self.train_ratio)
            remain = N - train_cnt
            val_cnt = int(remain * self.val_ratio)
            test_cnt = remain - val_cnt
            if self.flag == 'train':
                sub_df = gdf.iloc[:train_cnt]
            elif self.flag == 'val':
                sub_df = gdf.iloc[train_cnt: train_cnt + val_cnt]
            elif self.flag == 'test':
                sub_df = gdf.iloc[train_cnt + val_cnt: train_cnt + val_cnt + test_cnt]
            else:
                raise ValueError
            if len(sub_df) < (self.seq_len + self.pred_len):
                continue
            sub_dfs.append((zipcode, sub_df))
        if self.scale:
            if self.flag == 'train':
                scaler = StandardScaler()
                for _, sdf in sub_dfs:
                    scaler.partial_fit(sdf[enc_cols].values)
                Dataset_Custom._shared_scaler = scaler
            else:
                scaler = Dataset_Custom._shared_scaler
                if scaler is None:
                    raise RuntimeError
        else:
            scaler = None
        for zipcode, sub_df in sub_dfs:
            X = sub_df[enc_cols].values
            if self.features == 'M':
                Y = X.copy()
            else:
                Y = sub_df[[self.target]].values
            X_mark = sub_df[time_mark_cols].values
            if scaler is not None:
                X = scaler.transform(X)
            series_idx = len(self.series_list)
            self.series_list.append(X)
            self.out_list.append(Y)
            self.xmark_list.append(X_mark)
            length = X.shape[0]
            max_start = length - (self.seq_len + self.pred_len)
            for st in range(max_start + 1):
                self.index_map.append((series_idx, st))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        series_idx, st = self.index_map[idx]
        X = self.series_list[series_idx]
        Y = self.out_list[series_idx]
        X_mark = self.xmark_list[series_idx]
        seq_x = X[st: st + self.seq_len]
        seq_x_mark = X_mark[st: st + self.seq_len]
        r_begin = (st + self.seq_len) - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_y = Y[r_begin: r_end]
        seq_y_mark = X_mark[r_begin: r_end]
        seq_x = torch.from_numpy(seq_x).float()
        seq_y = torch.from_numpy(seq_y).float()
        seq_x_mark = torch.from_numpy(seq_x_mark).float()
        seq_y_mark = torch.from_numpy(seq_y_mark).float()
        return seq_x, seq_y, seq_x_mark, seq_y_mark
