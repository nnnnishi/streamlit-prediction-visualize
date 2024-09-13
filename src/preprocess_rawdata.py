import pandas as pd


def load_sushi_data(file_path):
    """
    寿司のスコアデータを読み込む関数
    Parameters
    ----------
    file_path : str
        ファイルのパス
    Returns
    ----------
    pd.DataFrame
        寿司のスコアデータ
    """
    # ファイルを読み込み、列名を設定
    data = pd.read_csv(file_path, delimiter=" ", header=None)
    data.columns = ["item_" + str(i) for i in range(100)]
    data.insert(0, "user_id", range(len(data)))
    return data


def load_sushi_items(file_path):
    """
    寿司のアイテムデータを読み込む関数
    Parameters
    ----------
    file_path : str
        ファイルのパス
    Returns
    ----------
    pd.DataFrame
        寿司のアイテムデータ
    """

    # ファイルを読み込み、列名を設定
    items = pd.read_csv(file_path, delimiter="\t", header=None, encoding="utf-8")
    items.columns = [
        "item_id",
        "name",
        "style",
        "major_group",
        "minor_group",
        "oiliness",
        "eating_frequency",
        "price",
        "selling_frequency",
        "name_ja",
    ]
    return items


def preprocess_sushi_data(sushi_data, sushi_items):
    """
    寿司データを整形する関数
    Parameters
    ----------
    sushi_data : pd.DataFrame
        寿司のスコアデータ
    sushi_items : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        整形後の寿司データ
    """
    # melt関数を使ってスコアデータを横持ちから縦持ちに変換
    # meltの説明: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html
    # meltのイメージ
    #   user_id item_0 item_1 item_2
    # 0       0      5      4      3
    # 1       1      3      2      1
    # ↓
    #   user_id item_id score
    # 0       0       0     5
    # 1       0       1     4
    # 2       0       2     3
    # 3       1       0     3
    # 4       1       1     2
    # 5       1       2     1
    melted_data = sushi_data.melt(
        id_vars=["user_id"], var_name="item_id", value_name="score"
    )
    # item_idの数字部分のみを抽出し、整数型に変換
    melted_data["item_id"] = melted_data["item_id"].str.extract(r"(\d+)").astype(int)
    # スコアが-1のデータを削除
    melted_data = melted_data[melted_data["score"] != -1]
    # item_idに対応する寿司の名前を結合
    melted_data = melted_data.merge(
        sushi_items[["item_id", "name", "name_ja"]], on="item_id", how="left"
    )
    # ユーザーごとにランク付け、rankは整数値にする
    # rankの説明: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html
    melted_data["rank"] = (
        melted_data.groupby("user_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    # 必要な列だけを選択し、ソート
    result = melted_data[["user_id", "rank", "item_id", "name", "name_ja", "score"]]
    result = result.sort_values(["user_id", "rank"])
    return result


score_file_path = "rawdata/sushi3-2016/sushi3b.5000.10.score"
items_file_path = "rawdata/sushi3-2016/sushi3_2.idata"

# スコアデータの読み込み
sushi_data = load_sushi_data(score_file_path)
# アイテムデータの読み込み
sushi_items = load_sushi_items(items_file_path)
# データの整形
preprocessed_data = preprocess_sushi_data(sushi_data, sushi_items)

# CSVファイルに出力
preprocessed_data.to_csv("data/sushi_ratings.csv", index=False)
sushi_items.to_csv("data/sushi_items.csv", index=False)
