
from pandas import DataFrame, to_datetime, concat
qp = lambda p: {'name': f'q{p}'.replace('0.',''), 'func': lambda x: x.quantile(q=p)}

def feature_groupby(
    data: DataFrame,
    group_column: str,
    column: list,
    agg_func: str
    ) -> DataFrame:
  all_column = [group_column, *column]
  df = data[all_column].copy()
  if type(agg_func) == dict:
    new_name = [f'{group_column}_{col}_{agg_func["name"]}' for col in column]
    df[new_name] = df.groupby([group_column])[column].transform(agg_func['func'])
  elif type(agg_func) == str:
    new_name = [f'{group_column}_{col}_{agg_func}' for col in column]
    df[new_name] = df.groupby([group_column])[column].transform(agg_func)
  return df.drop(all_column, axis=1)
  
def timestamp_feature(df: DataFrame) -> DataFrame:
  df.timestamp = to_datetime(df.timestamp)
  df['week'] = df.timestamp.dt.isocalendar().week
  df['day_of_week'] = df.timestamp.dt.weekday
  df['hour'] = df.timestamp.dt.hour

  mode_user = feature_groupby(
      df, 'user_id', ['hour', 'day_of_week', 'week'], {'name': 'mode', 'func':lambda x: x.mode()[0]}
      )
  mode_item = feature_groupby(
      df, 'item_id', ['hour', 'day_of_week', 'week'], {'name': 'mode', 'func':lambda x: x.mode()[0]}
      )
  df = concat([df, mode_user, mode_item], axis=1)

  agg_arr = ['max', 'mean']
  for agg_func in agg_arr:
    agg_user = feature_groupby(df, 'user_id', ['hour', 'day_of_week'], agg_func)
    agg_item = feature_groupby(df, 'item_id', ['hour', 'day_of_week'], agg_func)
    df = concat([df, agg_user, agg_item], axis=1)
  return df.drop(['hour', 'day_of_week', 'week'], axis=1)

def timestamp_feature(df: DataFrame, df_pred: DataFrame) -> DataFrame:
  df.timestamp = to_datetime(df.timestamp)
  df['week'] = df.timestamp.dt.isocalendar().week
  df['day_of_week'] = df.timestamp.dt.weekday
  df['hour'] = df.timestamp.dt.hour

  mode_user = feature_groupby(
      df, 'user_id', ['hour', 'day_of_week', 'week'], {'name': 'mode', 'func':lambda x: x.mode()[0]}
      )
  mode_item = feature_groupby(
      df, 'item_id', ['hour', 'day_of_week', 'week'], {'name': 'mode', 'func':lambda x: x.mode()[0]}
      )
  df = concat([df, mode_user, mode_item], axis=1)

  agg_arr = ['max', 'mean']
  for agg_func in agg_arr:
    agg_user = feature_groupby(df, 'user_id', ['hour', 'day_of_week'], agg_func)
    agg_item = feature_groupby(df, 'item_id', ['hour', 'day_of_week'], agg_func)
    df = concat([df, agg_user, agg_item], axis=1)
  return df.drop(['hour', 'day_of_week', 'week'], axis=1)

def rating_feature(df: DataFrame) -> DataFrame:
  group_columns = ['user_id', 'item_id']
  agg_arr = ['max', 'mean', qp(0.90), qp(0.95), qp(0.99)]
  for agg_func in agg_arr:
    for group in group_columns:
      agg_ = feature_groupby(df, group, ['rating'], agg_func)
      df = concat([df, agg_], axis=1)
  return df

def freq_item_feature(df: DataFrame) -> DataFrame:
  freq_item = df['item_id'].value_counts(normalize=True).rename('freq_item')
  df = df.join(freq_item, on = 'item_id', how = 'inner')

  agg_arr = ['max', 'mean', 'median', qp(0.25), qp(0.75)]
  for agg_func in agg_arr:
    agg_ = feature_groupby(df, 'user_id', ['freq_item'], agg_func)
    df = concat([df, agg_], axis=1)
  return df