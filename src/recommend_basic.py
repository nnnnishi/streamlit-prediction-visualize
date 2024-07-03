# 寿司データセットを用いた協調フィルタリングによるレコメンデーションの実装
import pandas as pd
from surprise import Dataset, Reader, SVD, NMF
from surprise.model_selection import train_test_split
from collections import defaultdict


# データの読み込みと前処理
def load_sushi_data(file_path):
    data = pd.read_csv(file_path, delimiter=" ", header=None)
    # ユーザーIDの列をデータフレームに追加
    data.reset_index(inplace=True)
    data.columns = ["user_id"] + [f"item_{i}" for i in range(100)]
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


def preprocess_data(data):
    melted_data = data.melt(
        id_vars=["user_id"], var_name="item_id", value_name="rating"
    )
    melted_data["item_id"] = melted_data["item_id"].str.extract("(\d+)").astype(int)
    melted_data = melted_data[melted_data["rating"] != -1]
    return melted_data


# 協調フィルタリングモデルの訓練と予測, SVDとNMFの両方を試す
def train_and_predict_models(data):
    reader = Reader(rating_scale=(0, 4))
    dataset = Dataset.load_from_df(data[["user_id", "item_id", "rating"]], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

    models = {"svd": SVD(), "nmf": NMF()}

    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(trainset)
        print(f"Training {name} model...done")

    return models


# 未評価アイテムの予測
def get_top_n_recommendations(model, data, items_df, n=5):
    user_items = defaultdict(set)
    for _, row in data.iterrows():
        user_items[row["user_id"]].add(row["item_id"])

    top_n = defaultdict(list)
    for user_id in user_items.keys():
        for item_id in range(100):
            if item_id not in user_items[user_id]:
                predicted_rating = model.predict(user_id, item_id).est
                item_name = items_df.loc[item_id, "name"]
                top_n[user_id].append((item_id, item_name, predicted_rating))

    for user_id, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[user_id] = user_ratings[:n]

    return top_n


# 結果の保存
def process_and_save_results(models, processed_data, sushi_items, n=5):
    for model_name, model in models.items():
        top_n_recommendations = get_top_n_recommendations(
            model, processed_data, sushi_items, n
        )

        results = []
        for user_id, recommendations in top_n_recommendations.items():
            for rank, (item_id, item_name, predicted_rating) in enumerate(
                recommendations, 1
            ):
                results.append(
                    {
                        "user_id": user_id,
                        "rank": rank,
                        "item_id": item_id,
                        "item_name": item_name,
                        "predicted_rating": predicted_rating,
                    }
                )

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by=["user_id", "rank"])
        df_results.to_csv(
            f"recommend_score/{model_name}_results.csv",
            index=False,
            float_format="%.2f",
        )


# メイン処理
file_path = "rawdata/sushi3-2016/sushi3b.5000.10.score"
items_file_path = "rawdata/sushi3-2016/sushi3.idata"
sushi_data = load_sushi_data(file_path)
sushi_items = load_sushi_items(items_file_path)
processed_data = preprocess_data(sushi_data)
models = train_and_predict_models(processed_data)
process_and_save_results(models, processed_data, sushi_items)
