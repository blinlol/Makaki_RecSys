from typing import Literal
import pandas as pd
from pandas import DataFrame, to_datetime, concat

qp = lambda p: {"name": f"q{p}".replace("0.", ""), "func": lambda x: x.quantile(q=p)}


def add_sequence_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column with the length of the sequence for each user."""
    df["sequence_length"] = df.groupby("user_id")["user_id"].transform("count")
    return df


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Колонка с ранком предмета из СаСрека"""
    df = df.sort_values("prediction", ascending=False)
    df["rank"] = df.groupby("user_id").cumcount() + 1
    return df


def feature_groupby(
    data: DataFrame,
    group_column: Literal["user_id", "item_id"],
    column: list,
    agg_func: str | dict,
) -> DataFrame:
    all_column = [group_column, *column]
    df = data[all_column].copy()
    if isinstance(agg_func, dict):
        new_name = [f'{group_column}_{col}_{agg_func["name"]}' for col in column]
        df[new_name] = df.groupby([group_column])[column].transform(agg_func["func"])
    elif isinstance(agg_func, str):
        new_name = [f"{group_column}_{col}_{agg_func}" for col in column]
        df[new_name] = df.groupby([group_column])[column].transform(agg_func)
    return df.drop(all_column, axis=1)


def timestamp_feature(df: DataFrame, group: Literal["user_id", "item_id"]) -> DataFrame:
    df.timestamp = to_datetime(df.timestamp)
    df["week"] = df.timestamp.dt.isocalendar().week
    df["day_of_week"] = df.timestamp.dt.weekday
    df["hour"] = df.timestamp.dt.hour

    mode = feature_groupby(
        df,
        group,
        ["hour", "day_of_week", "week"],
        {"name": "mode", "func": lambda x: x.mode()[0]},
    )

    df = concat([df, mode], axis=1)

    agg_arr = ["max", "mean"]
    for agg_func in agg_arr:
        agg = feature_groupby(df, group, ["hour", "day_of_week"], agg_func)
        df = concat([df, agg], axis=1)
    return df.drop(["hour", "day_of_week", "week"], axis=1)


def rating_feature(df: DataFrame, group: Literal["user_id", "item_id"]) -> DataFrame:
    agg_arr = ["max", "mean", qp(0.90), qp(0.95), qp(0.99)]
    for agg_func in agg_arr:
        agg_ = feature_groupby(df, group, ["rating"], agg_func)
        df = concat([df, agg_], axis=1)
    return df


def freq_item(df: DataFrame) -> DataFrame:
    freq_item = df["item_id"].value_counts(normalize=True).rename("freq_item")
    return df.join(freq_item, on="item_id", how="inner")


def freq_user_item(df: DataFrame) -> DataFrame:
    agg_arr = ["max", "mean", "median", qp(0.25), qp(0.75)]
    for agg_func in agg_arr:
        agg_ = feature_groupby(df, "user_id", ["freq_item"], agg_func)
        df = concat([df, agg_], axis=1)
    return df


def make_features(pred_data: pd.DataFrame, data: pd.DataFrame, train_path: str) -> pd.DataFrame:
    train_data = pd.read_parquet(train_path)

    train_data = add_sequence_length(train_data)
    pred_data = pred_data.merge(
        train_data[["user_id", "sequence_length"]].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )
    data = data.merge(
        train_data[["user_id", "sequence_length"]].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )
    train_data = train_data.drop(columns=["sequence_length"])

    train_data = timestamp_feature(train_data, "item_id")
    pred_data = pred_data.merge(
        train_data[
            [
                "item_id",
                "item_id_hour_mode",
                "item_id_day_of_week_mode",
                "item_id_week_mode",
                "item_id_hour_max",
                "item_id_day_of_week_max",
                "item_id_hour_mean",
                "item_id_day_of_week_mean",
            ]
        ].drop_duplicates(subset=["item_id"]),
        on="item_id",
        how="left",
    )
    data = data.merge(
        train_data[
            [
                "item_id",
                "item_id_hour_mode",
                "item_id_day_of_week_mode",
                "item_id_week_mode",
                "item_id_hour_max",
                "item_id_day_of_week_max",
                "item_id_hour_mean",
                "item_id_day_of_week_mean",
            ]
        ].drop_duplicates(subset=["item_id"]),
        on="item_id",
        how="left",
    )
    
    train_data = train_data.drop(
        columns=[
            "item_id_hour_mode",
            "item_id_day_of_week_mode",
            "item_id_week_mode",
            "item_id_hour_max",
            "item_id_day_of_week_max",
            "item_id_hour_mean",
            "item_id_day_of_week_mean",
        ]
    )

    train_data = timestamp_feature(train_data, "user_id")
    data = data.merge(
        train_data[
            [
                "user_id",
                "user_id_hour_mode",
                "user_id_day_of_week_mode",
                "user_id_week_mode",
                "user_id_hour_max",
                "user_id_day_of_week_max",
                "user_id_hour_mean",
                "user_id_day_of_week_mean",
            ]
        ].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )
    pred_data = pred_data.merge(
        train_data[
            [
                "user_id",
                "user_id_hour_mode",
                "user_id_day_of_week_mode",
                "user_id_week_mode",
                "user_id_hour_max",
                "user_id_day_of_week_max",
                "user_id_hour_mean",
                "user_id_day_of_week_mean",
            ]
        ].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )
    train_data = train_data.drop(
        columns=[
            "user_id_hour_mode",
            "user_id_day_of_week_mode",
            "user_id_week_mode",
            "user_id_hour_max",
            "user_id_day_of_week_max",
            "user_id_hour_mean",
            "user_id_day_of_week_mean",
        ]
    )

    train_data = rating_feature(train_data, "item_id")

    pred_data = pred_data.merge(
        train_data[
            [
                "item_id",
                "item_id_rating_max",
                "item_id_rating_mean",
                "item_id_rating_q9",
                "item_id_rating_q95",
                "item_id_rating_q99",
            ]
        ].drop_duplicates(subset=["item_id"]),
        on="item_id",
        how="left",
    )
    data = data.merge(
        train_data[
            [
                "item_id",
                "item_id_rating_max",
                "item_id_rating_mean",
                "item_id_rating_q9",
                "item_id_rating_q95",
                "item_id_rating_q99",
            ]
        ].drop_duplicates(subset=["item_id"]),
        on="item_id",
        how="left",
    )
    train_data = train_data.drop(
        columns=[
            "item_id_rating_max",
            "item_id_rating_mean",
            "item_id_rating_q9",
            "item_id_rating_q95",
            "item_id_rating_q99",
        ]
    )

    train_data = rating_feature(train_data, "user_id")
    pred_data = pred_data.merge(
        train_data[
            [
                "user_id",
                "user_id_rating_max",
                "user_id_rating_mean",
                "user_id_rating_q9",
                "user_id_rating_q95",
                "user_id_rating_q99",
            ]
        ].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )
    data = data.merge(
        train_data[
            [
                "user_id",
                "user_id_rating_max",
                "user_id_rating_mean",
                "user_id_rating_q9",
                "user_id_rating_q95",
                "user_id_rating_q99",
            ]
        ].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )
    train_data = train_data.drop(
        columns=[
            "user_id_rating_max",
            "user_id_rating_mean",
            "user_id_rating_q9",
            "user_id_rating_q95",
            "user_id_rating_q99",
        ]
    )

    train_data = freq_item(train_data)
    pred_data = pred_data.merge(
        train_data[["item_id", "freq_item"]].drop_duplicates(subset=["item_id"]),
        on="item_id",
        how="left",
    )
    data = data.merge(
        train_data[["item_id", "freq_item"]].drop_duplicates(subset=["item_id"]),
        on="item_id",
        how="left",
    )
    train_data.drop(columns=["freq_item"])

    train_data = freq_user_item(train_data)
    pred_data = pred_data.merge(
        train_data[
            [
                "user_id",
                "user_id_freq_item_max",
                "user_id_freq_item_mean",
                "user_id_freq_item_median",
                "user_id_freq_item_q25",
                "user_id_freq_item_q75",
            ]
        ].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )
    data = data.merge(
        train_data[
            [
                "user_id",
                "user_id_freq_item_max",
                "user_id_freq_item_mean",
                "user_id_freq_item_median",
                "user_id_freq_item_q25",
                "user_id_freq_item_q75",
            ]
        ].drop_duplicates(subset=["user_id"]),
        on="user_id",
        how="left",
    )
    del train_data

    return pred_data, data


def make_features_after(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rank(df)
    return df
