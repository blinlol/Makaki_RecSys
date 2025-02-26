# -*- coding: utf-8 -*-
"""feature_func.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gRJ6Ea7QRsRZJpRpr8_QMPEVBh4fIrJ3
"""

import pandas as pd

def freq_items(data):
  freqs = data.groupby(['item_id'])['user_id'].count()
  freqs = freqs / freqs.count()
  freqs = freqs.rename('freq_item')

  data_ = data.join(freqs, on = 'item_id', how = 'inner')
  return data_

#перед использованием надо выпонить функцию freq_items
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

  stats = pd.concat([mean, quantile_25, quantile_50, quantile_75, max], axis=1)

  data_ = data.join(stats, on = 'user_id', how = 'inner')
  return data_

def rating_item(data):
  mean = data.groupby(['item_id'])['rating'].mean()
  mean = mean.rename('rating_item_mean')

  max = data.groupby(['item_id'])['rating'].max()
  max = max.rename('rating_item_max')

  quantile_90 = data.groupby(['item_id'])['rating'].quantile(q = .90)
  quantile_90 = quantile_90.rename('rating_item_quantile_90')

  quantile_95 = data.groupby(['item_id'])['rating'].quantile(q = .95)
  quantile_95 = quantile_95.rename('rating_item_quantile_95')

  quantile_99 = data.groupby(['item_id'])['rating'].quantile(q = .99)
  quantile_99 = quantile_99.rename('rating_item_quantile_99')

  stats = pd.concat([mean, quantile_90, quantile_95, quantile_99, max], axis=1)

  data_ = data.join(stats, on = 'item_id', how = 'inner')
  return data_

def rating_user(data):
  mean = data.groupby(['user_id'])['rating'].mean()
  mean = mean.rename('rating_user_mean')

  max = data.groupby(['user_id'])['rating'].max()
  max = max.rename('rating_user_max')

  quantile_90 = data.groupby(['user_id'])['rating'].quantile(q = .90)
  quantile_90 = quantile_90.rename('rating_user_quantile_90')

  quantile_95 = data.groupby(['user_id'])['rating'].quantile(q = .95)
  quantile_95 = quantile_95.rename('rating_user_quantile_95')

  quantile_99 = data.groupby(['user_id'])['rating'].quantile(q = .99)
  quantile_99 = quantile_99.rename('rating_user_quantile_99')

  stats = pd.concat([mean, quantile_90, quantile_95, quantile_99, max], axis=1)

  data_ = data.join(stats, on = 'user_id', how = 'inner')
  return data_