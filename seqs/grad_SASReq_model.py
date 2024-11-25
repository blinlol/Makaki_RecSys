import os
import time
import sys
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import random
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from torch.utils.data import DataLoader

from datasets import (CausalLMDataset, CausalLMPredictionDataset,
                         PaddingCollateFn)
from models import SASRec, GRU4Rec
from modules import SeqRec, SeqRecWithSampling
from postprocess import preds2recs
from split import *




# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['PYTHONPATH'] = ""
# sys.path.remove('/home/jovyan/.imgenv-vasilyev-0/lib/python3.7/site-packages')


VAl_USER = 500
SAMPLE_SIZE = 80000




def get_top_k(data: pd.DataFrame, k=10) -> list[int]:
    top = {}
    for item_id in data.item_id:
        top[item_id] = top.get(item_id, 0) + 1
    if k == -1:
        list(map(lambda e: e[0], sorted(top.items(), reverse=True, key=lambda e: e[1])))
    return list(map(lambda e: e[0], sorted(top.items(), reverse=True, key=lambda e: e[1])[:k]))


def fit_predict(dataset, top_k=10):
    test_fname = f"test_{dataset}.parquet"
    train_fname = f"train_{dataset}.parquet"

    data = pd.read_parquet(train_fname)

    train = preprocessing(data, item_min_count=5, min_len=5)

    val_user = np.random.choice(train.user_id.unique(), int(VAl_USER))
    validation = train[train["user_id"].isin(val_user)]

    train = train[~train["user_id"].isin(val_user)]
    max_item_id = data.item_id.max()

    sampled_users = np.random.choice(train['user_id'].unique(),
                                            size=SAMPLE_SIZE, replace=False)
    filtered_train = train[train["user_id"].isin(sampled_users)]

    train_loader, eval_loader = create_dataloaders(filtered_train, validation)

    model = create_model(item_count=max_item_id)
    start_time = time.time()
    trainer, seqrec_module, model = training(model, train_loader, eval_loader)
    training_time = time.time() - start_time
    print('training_time', training_time)

    seqrec_module.filter_seen = True


    data = pd.read_parquet(test_fname)
    return predict(trainer, seqrec_module, data, top_k=top_k), trainer, seqrec_module, model
    # data_hot, cold_users = filter_users_with_few_interactions(data)
    # recs_hot = predict(trainer, seqrec_module, data_hot, top_k=10)
    # recs_cold = predict_cold_users(data, get_top_k(data, -1), cold_users)
    # return pd.concat([recs_hot, recs_cold], axis=0, ignore_index=True)


def filter_users_with_few_interactions(df):
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
        # result_df.append({'user_id': user_id, 'item_id': pred_items})
        result_df.extend([{'user_id': user_id, 'item_id': item, 'prediction': ind}
                                for ind, item in enumerate(pred_items[::-1])])

    return pd.DataFrame(result_df)


def create_dataloaders(train, validation):

    train_dataset = CausalLMDataset(train, max_length=128)
    eval_dataset = CausalLMPredictionDataset(
                    validation, max_length=128, validation_mode=True, )

    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=8,
                              collate_fn=PaddingCollateFn())
    eval_loader = DataLoader(eval_dataset, batch_size=256,
                             shuffle=False, num_workers=8,
                             collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def create_model(item_count):

    model = SASRec(item_num=item_count, hidden_units=128, num_blocks=2,
                   num_heads=2, dropout_rate=0.1)

    return model


def training(model, train_loader, eval_loader):

    seqrec_module = SeqRec(model, lr=0.001, predict_top_k=10, filter_seen=False)

    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=5, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                 mode="max", save_weights_only=True)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks=[early_stopping, model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True, max_epochs=100)

    trainer.fit(model=seqrec_module,
            train_dataloaders=train_loader,
            val_dataloaders=eval_loader)

    seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

    return trainer, seqrec_module, model


def predict(trainer, seqrec_module, data, top_k=10):

    predict_dataset = CausalLMPredictionDataset(
        data, max_length=128)

    predict_loader = DataLoader(
        predict_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=256,
        num_workers=8)

    seqrec_module.predict_top_k = top_k
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)

    recs = preds2recs(preds)
    print('recs shape', recs.shape)

    return recs


def evaluate(recs, test_last, topk=[10], prefix='test'):

    all_metrics = {}

    for k in topk:
        evaluator = Evaluator(top_k=[k])
        metrics = evaluator.compute_metrics(test_last, recs)
        metrics = {prefix + '_' + key: value for key, value in metrics.items()}
        all_metrics.update(metrics)
    print(all_metrics)
    return all_metrics
