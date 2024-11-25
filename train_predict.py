import argparse
import os
from pathlib import Path
from typing import Tuple
import pandas as pd

import SASReq_model as sus

from ALS import ALSModel


cfg_data = {
    "user_column": "user_id",
    "item_column": "item_id",
    "date_column": "timestamp",
    "rating_column": "weight",
    "weighted": False,
    "dataset_names": ["smm", "zvuk"],
    "data_dir": "./",
    "model_dir": "./saved_models",
}


def SusRec_to_answer_df(pred: pd.DataFrame, k=10) -> pd.DataFrame:
    d = {
        'user_id': [],
        'item_id': [],
    }
    for uid in pred.user_id.unique():
        sorted = pred[pred.user_id == uid].sort_values('prediction', ascending=False)
        top_k = sorted.item_id.to_list()[:k]

        d['user_id'].append(uid)
        d['item_id'].append(top_k)
    return pd.DataFrame(d)


def zvuk():
    fname = "test_zvuk.parquet"
    reqs = sus.fit_predict(fname)
    ans = SusRec_to_answer_df(reqs)
    pred_path = Path(cfg_data["data_dir"]) / f"submission_zvuk.parquet"
    ans.to_parquet(pred_path)


def smm():
    model = fit()
    predict_smm(model)


class ModelTopK:
    def __init__(self, n):
        self.n = n
        self.smm_top_k = []
        self.smm_model = None
        self.zvuk_top_k = []
        self.indices_info = None


def get_top_k(data: pd.DataFrame, k=10) -> list[int]:
    top = {}
    for item_id in data.item_id:
        top[item_id] = top.get(item_id, 0) + 1
    if k == -1:
        list(map(lambda e: e[0], sorted(top.items(), reverse=True, key=lambda e: e[1])))
    return list(map(lambda e: e[0], sorted(top.items(), reverse=True, key=lambda e: e[1])[:k]))


def fit() -> None:
    smm_path = os.path.join(cfg_data["data_dir"], "train_smm.parquet")
    zvuk_path = os.path.join(cfg_data["data_dir"], "train_zvuk.parquet")
    print("Train smm-events:", smm_path)
    print("Train zvuk-events:", zvuk_path)
    smm_events = pd.read_parquet(smm_path)
    
    train_events = smm_events
    train_events["weight"] = 1
    
    smm_model = ALSModel(
        cfg_data,
        factors=200,
        regularization=0.002,
        iterations=200,
        alpha=20,
    )
    smm_model.fit(train_events)

    model = ModelTopK(10)
    model.smm_model = smm_model
    model.smm_top_k = get_top_k(train_events, -1)

    return model
    
def predict_smm(model):
    test_data = pd.read_parquet(os.path.join(cfg_data["data_dir"], f"test_smm.parquet"))
    test_data["weight"] = 1
    recs, user_ids = model.smm_model.recommend_k(test_data, k=100)
    
    recs = pd.Series([np.array(x) for x in recs.tolist()], index=user_ids)
    recs = recs.reset_index()
    recs.columns = ["user_id", "item_id"]

    recs = filter_smm(test_data, recs, model.smm_top_k)
    
    prediction_path = Path(cfg_data["data_dir"]) / f"submission_smm.parquet"
    recs.to_parquet(prediction_path)


def filter_smm(test_smm, recs, top_items):
    result_df = []
    smm_set = test_smm.groupby('user_id')['item_id'].apply(set).reset_index()

    for user_id in recs['user_id'].unique():
        user_items = recs[recs['user_id'] == user_id]['item_id'].iloc[0]
        pred_items = []
        for item_id in user_items:
            if item_id not in smm_set[smm_set['user_id'] == user_id]['item_id'].iloc[0]:
                pred_items.append(item_id)
            if len(pred_items) == 10:
                break
        if len(pred_items) != 10:
            i = 0
            while len(pred_items) != 10:
                pred_items.append(top_items[i])
                i += 1
        
        result_df.append({'user_id': user_id, 'item_id': pred_items})
    return pd.DataFrame(result_df)



def main():
    zvuk()
    smm()
    

if __name__ == "__main__":
    main()
