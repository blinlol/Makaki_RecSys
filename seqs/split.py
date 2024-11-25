import numpy as np
import pandas as pd


def preprocessing(data, users_sample=None, item_min_count=5, min_len=5, user_id='user_id', item_id='item_id',                                     timestamp='timestamp'):

    step = 1
    while (data['user_id'].value_counts().min() < min_len or data['item_id'].value_counts().min() < item_min_count):

        print(f'n-core filtering step {step}')
        data = drop_short_sequences(data, min_len)
        data = filter_items(data, item_min_count)

    return data


def filter_items(data, item_min_count, item_id="item_id"):
    """Filter items by occurrence threshold."""

    counts = data[item_id].value_counts()
    data = data[data[item_id].isin(counts[counts >= item_min_count].index)]

    return data



def drop_short_sequences(data, min_len, user_id='user_id'):
    """Drop user sequences shorter than given threshold."""

    counts = data[user_id].value_counts()
    users = counts[counts >= min_len].index
    data = data[data[user_id].isin(users)]

    return data

    
def leave_one_out_split(data, validation_size=500, user_id='user_id', item_id='item_id',
                        timestamp='timestamp'):
    
    data.sort_values([user_id, timestamp], inplace=True)
    data['time_idx_reversed'] = data.groupby(user_id).cumcount(ascending=False)
    
    train = data[data.time_idx_reversed >= 1]
    test = data
    val_user = np.random.choice(train.user_id.unique(), int(validation_size))
    validation = train[train[user_id].isin(val_user)]
    train = train[~train[user_id].isin(val_user)]
    
    return train, validation, test


def remove_last_item(data, user_id='user_id', item_id='item_id', timestamp='timestamp'):
    """Remove last item from each user sequence."""

    data.sort_values([user_id, timestamp], inplace=True)
    short_data = data.groupby(user_id)[item_id].agg(list).apply(
        lambda x: x[:-1]).reset_index().explode(item_id)
    short_data[timestamp] = data.groupby(user_id)[timestamp].agg(list).apply(
        lambda x: x[:-1]).reset_index().explode(timestamp)[timestamp]

    return short_data


def get_last_item(data, user_id='user_id', item_id='item_id', timestamp='timestamp'):
    """Get last item from each user sequence."""

    data.sort_values([user_id, timestamp], inplace=True)
    data_last = data.groupby(user_id)[item_id].agg(list).apply(lambda x: x[-1]).reset_index()

    return data_last