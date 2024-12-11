import os
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple
import io

import numpy as np
import pandas as pd
import torch

from src.model import ConvVAE
import xgboost as xgb


@dataclass
class PlantStat:
    """ 식물 생장/수확 데이터를 저장하는 데이터 클래스
    """
    # 초장 길이
    length: float
    # 누적 착과수
    num_fruit: float
    # 누적 수확수
    num_harvest: float
    # 누적 수확중량
    weight: float


class GrowthPredictor:
    def __init__(self, data_dir: str):
        self.model = load_embed_model(os.path.join(data_dir, 'embed'))
        self.regression_models = load_regression_models(os.path.join(data_dir, 'regression'))

    def predict(self, stat: PlantStat, uploaded_file: io.BytesIO) -> pd.DataFrame:
        """ 식물의 생육/수확 데이터를 예측하는 함수

        Args:
            stat (PlantStat): 식물의 생장/수확 데이터
            uploaded_file (str): 시계열 데이터 파일
        """
        # 1. 시계열 데이터 임베딩
        ts_df = pd.read_csv(uploaded_file)
        timeseries_data = self.embed_timeseries_data(ts_df)
        start_dt = timeseries_data[0][0]

        outputs = [(start_dt, stat.length, stat.num_fruit, stat.num_harvest, stat.weight)]
        for start_dt, embed_arr in timeseries_data:
            end_dt = start_dt + timedelta(days=7)
            # 2. 재귀적으로 처리
            stat = self.infer_step(embed_arr, stat)
            outputs.append((end_dt, stat.length, stat.num_fruit, stat.num_harvest, stat.weight))

        # 3. 결과 반환
        output_df = pd.DataFrame(outputs, columns=["날짜", "초장", "착과수", "누적수확수", "누적수확중량"])
        output_df = output_df.set_index('날짜')
        return output_df

    def embed_timeseries_data(self, data: pd.DataFrame, window_size=168) -> List[Tuple[date, np.ndarray]]:
        """ 시계열 데이터를 임베딩하기

        Args:
            data (pd.DataFrame): 전처리할 데이터
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

        # window_size 만큼 rolling 후, time-series를 임베딩
        result = []
        self.model.eval()
        with torch.no_grad():
            for window in data.rolling(window_size, step=window_size):
                if len(window) < window_size:
                    continue
                start_dt = window.index.min().date()
                window_arr = torch.tensor(window.values.T[None], dtype=torch.float32)
                embed_value = self.model.embed(window_arr)[0].detach().numpy()
                result.append((start_dt, embed_value))
        return result

    def infer_step(self, embed_value: np.ndarray, stat: PlantStat):
        """ 다음 스텝의 생장/수확 데이터를 예측
        """
        inputs = np.concatenate([
            np.array([stat.length, stat.num_fruit, stat.num_harvest, stat.weight, False]),
            embed_value
        ])[None]

        diff_length = self.regression_models['length'].predict(inputs)[0]
        diff_fruit = self.regression_models['fruit'].predict(inputs)[0]
        diff_harvest = self.regression_models['harvest'].predict(inputs)[0]
        # 총 수확중량이 아닌, 평균 수확중량을 예측 ( diff_harvest와 독립적인 변수로 예측 )
        diff_avg_weight = self.regression_models['weight'].predict(inputs)[0]

        stat.length += diff_length if diff_length > 0 else 0
        stat.num_fruit += diff_fruit if diff_fruit > 0 else 0
        stat.num_harvest += diff_harvest if diff_harvest > 0 else 0
        stat.weight += diff_avg_weight * diff_harvest if diff_avg_weight > 0 and diff_harvest > 0 else 0
        return stat


def load_embed_model(model_dir: str) -> ConvVAE:
    """ 임베딩 모델 불러오기 """
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        config = json.load(f)

    model = ConvVAE(
        num_steps=config['num_steps'],
        num_features=config['num_features'],
        num_hiddens=config['num_hiddens'],
        num_embeddings=config['num_embeddings']
    )

    model_path = os.path.join(model_dir, 'model.pt')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def load_regression_models(model_dir: str) -> Dict[str,xgb.XGBRegressor]:
    return {
        "length": load_xgb_regression_model(os.path.join(model_dir, "초장.ubj")),
        "fruit": load_xgb_regression_model(os.path.join(model_dir, "착과수.ubj")),
        "harvest": load_xgb_regression_model(os.path.join(model_dir, "수확수.ubj")),
        "weight": load_xgb_regression_model(os.path.join(model_dir, "평균과중.ubj")),
    }


def load_xgb_regression_model(model_path: str) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model
