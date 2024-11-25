import pandas as pd

def freq_items(data):
  freqs = data.groupby(['item_id'])['user_id'].count()
  freqs = freqs / freqs.count()
  freqs = freqs.rename('freq_item')

  data_ = data.join(freqs, on = 'item_id', how = 'inner')
  return data_

def freq_stats(data):
  mean = data.groupby(['user_id'])['freq_item'].mean()
  mean = mean.rename('freq_user_mean')

  max = data.groupby(['user_id'])['freq_item'].max()
  max = max.rename('freq_user_max')

  quantile_25 = data.groupby(['user_id'])['freq_item'].quantile(q = .25)
  quantile_25 = quantile_25.rename('freq_user_quantile_25')

  quantile_50 = data.groupby(['user_id'])['freq_item'].quantile(q = .5)
  quantile_50 = quantile_50.rename('freq_user_quantile_50')

  quantile_75 = data.groupby(['user_id'])['freq_item'].quantile(q = .75)
  quantile_75 = quantile_75.rename('freq_user_quantile_75')

  stats = pd.concat([mean, max, quantile_25, quantile_50, quantile_75], axis=1)

  data_ = data.join(stats, on = 'user_id', how = 'inner')
  return data_


def add_sequence_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column with the length of the sequence for each user."""
    df['sequence_length'] = df.groupby('user_id')['user_id'].transform('count')
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_sequence_length(df)
    df = freq_items(df)
    df = freq_stats(df)
    return df