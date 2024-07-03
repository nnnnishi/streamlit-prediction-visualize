# rawdataディレクトリにある寿司データを加工し、preprocessed_dataディレクトリに出力する
import pandas as pd


def load_sushi_data(file_path):
    data = pd.read_csv(file_path, delimiter=" ", header=None)
    data.columns = ["item_" + str(i) for i in range(100)]
    data.insert(0, "user_id", range(len(data)))
    return data


def load_sushi_items(file_path):
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
    ]
    return items


def process_sushi_data(sushi_data, sushi_items):
    melted_data = sushi_data.melt(
        id_vars=["user_id"], var_name="item_id", value_name="score"
    )
    melted_data["item_id"] = melted_data["item_id"].str.extract("(\d+)").astype(int)
    melted_data = melted_data[melted_data["score"] != -1]

    # アイテム名を追加
    melted_data = melted_data.merge(
        sushi_items[["item_id", "name"]], on="item_id", how="left"
    )

    # ユーザーごとにランク付け、rankは整数値にする
    melted_data["rank"] = (
        melted_data.groupby("user_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # 必要な列だけを選択し、ソート
    result = melted_data[["user_id", "rank", "item_id", "name", "score"]]
    result = result.sort_values(["user_id", "rank"])

    return result


# メイン処理
file_path = "rawdata/sushi3-2016/sushi3b.5000.10.score"
items_file_path = "rawdata/sushi3-2016/sushi3.idata"

sushi_data = load_sushi_data(file_path)
sushi_items = load_sushi_items(items_file_path)
processed_data = process_sushi_data(sushi_data, sushi_items)

# CSVファイルに出力
processed_data.to_csv("preprocessed_data/sushi_ratings.csv", index=False)

print("sushi_ratings.csv ファイルが作成されました。")
