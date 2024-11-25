import argparse
import os
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

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
    reqs = sus.fit_predict("zvuk")
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
        factors=128,
        regularization=0.002,
        iterations=200,
        alpha=10,
    )
    smm_model.fit(train_events)

    model = ModelTopK(10)
    model.smm_model = smm_model
    model.smm_top_k = get_top_k(train_events, -1)

    return model
    
def predict_smm(model):
    test_data = pd.read_parquet(os.path.join(cfg_data["data_dir"], f"test_smm.parquet"))
    test_data["weight"] = 1
    
    # hot_data, cold_users = filter_users_with_few_interactions(test_data)
    
    recs, user_ids = model.smm_model.recommend_k(test_data, k=100)
    
    recs = pd.Series([np.array(x) for x in recs.tolist()], index=user_ids)
    recs = recs.reset_index()
    recs.columns = ["user_id", "item_id"]

    recs = filter_smm(test_data, recs, model.smm_top_k)
    # cold_recs = predict_cold_users(test_data, model.smm_top_k, cold_users)
    # recs = pd.concat([hot_recs, cold_recs], axis=0, ignore_index=True)

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


def filter_users_with_few_interactions(df):
    """data_hot, cold_users"""
    user_interaction_counts = df['user_id'].value_counts()
    
    users_to_remove = user_interaction_counts[user_interaction_counts < 5].index.tolist()
    
    filtered_data = df[~df['user_id'].isin(users_to_remove)]
    
    return filtered_data, users_to_remove


def predict_cold_users(test, top_items, cold_users):
    print(top_items)

    sett = test.groupby('user_id')['item_id'].apply(set).reset_index()
    result_df = []
    # for user_id in test_zvuk['user_id'].unique():
    for user_id in cold_users:
        pred_items = []
        for item_id in top_items:
            if item_id not in sett[sett['user_id'] == user_id]['item_id'].iloc[0]:
                pred_items.append(item_id)
            if len(pred_items) == 10:
                break
        if len(pred_items) != 10:
            i = 0
            while len(pred_items) != 10:
                pred_items.append(top_items[i])
                i += 1
        result_df.append({'user_id': user_id, 'item_id': pred_items})
        # result_df.extend([{'user_id': user_id, 'item_id': item, 'prediction': ind}
        #                         for ind, item in enumerate(pred_items[::-1])])

    return pd.DataFrame(result_df)



def main():
    smm()
    zvuk()

    

if __name__ == "__main__":
    main()
