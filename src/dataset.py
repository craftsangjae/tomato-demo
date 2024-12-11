from typing import List, Tuple
from datetime import date, timedelta

import pandas as pd
import numpy as np
import torch
from src.model import ConvVAE


def embed_timeseries_data(data: pd.DataFrame, model: ConvVAE, window_size=168) -> List[Tuple[date, np.ndarray]]:
    """ 시계열 데이터를 임베딩하기

    Args:
        data (pd.DataFrame): 전처리할 데이터
        model (ConvVAE): 시계열 데이터를 임베딩하기 위한 모델
        window_size (int): 시계열 데이터를 임베딩하기 위한 window size, default=168 (1주일)
    """
    # 필수 컬럼이 여부 확인
    for neccessary_column in ["MSRMT_DT", "INR_TMPRTR", "INR_RH", "INR_CO2QY"]:
        if neccessary_column not in data.columns:
            raise ValueError(f"{neccessary_column} 컬럼이 존재하지 않습니다.")

    # 혹시 있을 결측치를 채워주기 위함
    data.loc[:, "MSRMT_DT"] = pd.to_datetime(data["MSRMT_DT"])
    data = data.set_index('MSRMT_DT').sort_index()
    data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq=timedelta(hours=1)))
    data = data.bfill()

    # TODO 데이터 전처리 코드 작성 필요

    # window_size 만큼 rolling 후, time-series를 임베딩
    result = []
    model.eval()
    with torch.no_grad():
        for window in data.rolling(window_size, step=window_size):
            if len(window) < window_size:
                continue
            start_dt = window.index.min().date()
            window_arr = torch.tensor(window.values.T[None], dtype=torch.float32)
            embed_value = model.embed(window_arr)[0].detach().numpy()
            result.append((start_dt, embed_value))
    return result
