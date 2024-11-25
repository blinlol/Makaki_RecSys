import argparse
import os
from pathlib import Path
from typing import Tuple, Literal
import pandas as pd
from make_features import make_features
from split import leave_one_out_split, get_last_item
import catboost as cb
import ALS
import numpy as np



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

    
def run_pipeline(dataset: Literal['zvuk', 'smm']):
    from grad_SASReq_model import fit_predict, predict    
    train_path = os.path.join(cfg_data["data_dir"], f"train_{dataset}.parquet")
    test_path = os.path.join(cfg_data["data_dir"], f"test_{dataset}.parquet")
        
    data, trainer, seqrec_module, _ = fit_predict(dataset) 
    _, _, test = leave_one_out_split(pd.read_parquet(test_path), validation_size=0)
    sasrec_out = predict(trainer, seqrec_module, test, top_k=100)
    
    train_data_featured, sasrec_out_featured = make_features(data, sasrec_out, train_path)
    
    model = fit_boosting(train_data_featured, test_path)
    
    result = predict_boosting(sasrec_out_featured, model)
    
    prediction_path = Path(cfg_data["data_dir"]) / f"submission_{dataset}.parquet"
    result.to_parquet(prediction_path)
    

def add_target(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    last_item = get_last_item(test)
        
    merged = train.merge(last_item, on=['user_id', 'item_id'], how='left', indicator=True)
    train['target'] = (merged['_merge'] == 'both').astype(int)
    
    return train

def fit_boosting(train, test_path):
    test = pd.read_parquet(test_path)
    
    train_with_target = add_target(train, test)
    
    model = cb.CatBoostClassifier(iterations=100, learning_rate=0.1, random_seed=42)
    model.fit(train_with_target.drop(columns=['target']), train_with_target['target'])
    return model


def predict_boosting(test, model):
    result = model.predict_proba(test)
    test['score'] = result[:, 0]
    output = test.groupby('user_id').apply(lambda x: x.sort_values('score', ascending=False).head(10)['item_id'].tolist())
    return pd.DataFrame({'user_id': output.index, 'item_id': output.values})    


def main():
    run_pipeline('zvuk')
    # run_pipeline('smm')
    smm()




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
    
    smm_model = ALS.ALSModel(
        cfg_data,
        factors=200,
        regularization=0.02,
        iterations=100,
        alpha=5,
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


    

if __name__ == "__main__":
    main()
