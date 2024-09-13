# conda install conda-forge::scikit-surprise が必要
import pandas as pd
from surprise import Dataset, Reader, SVD, NMF
from surprise.model_selection import train_test_split
from collections import defaultdict


# 協調フィルタリングモデルの訓練と予測, SVDとNMFの両方を試す
def train_and_predict_models(data):
    """
    協調フィルタリングモデルを訓練し、予測する関数
    Parameters
    ----------
    data : pd.DataFrame
        前処理済みのデータ
    Returns
    ----------
    dict
        訓練済みモデル
    """
    reader = Reader(rating_scale=(0, 4))
    dataset = Dataset.load_from_df(data[["user_id", "item_id", "score"]], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

    models = {"svd": SVD(), "nmf": NMF()}

    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(trainset)
        print(f"Training {name} model...done")

    return models


def get_top_n_recommendations(model, sushi_rating_df, sushi_items_df, n=5):
    """
    未評価アイテムの予測を取得する関数
    Parameters
    ----------
    model : object
        訓練済みモデル
    sushi_rating_df : pd.DataFrame
        評価データ
    sushi_items_df : pd.DataFrame
        アイテムデータ
    n : int
        予測するアイテム数
    Returns
    ----------
    dict
        予測結果
    """

    user_items = defaultdict(set)
    for _, row in sushi_rating_df.iterrows():
        user_items[row["user_id"]].add(row["item_id"])

    top_n = defaultdict(list)
    for user_id in user_items.keys():
        for item_id in range(100):
            if item_id not in user_items[user_id]:
                predicted_score = model.predict(user_id, item_id).est
                item_name = sushi_items_df.loc[item_id, "name"]
                top_n[user_id].append((item_id, item_name, predicted_score))

    for user_id, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[user_id] = user_ratings[:n]

    return top_n


# 結果の保存
def process_and_save_results(models, sushi_rating_df, sushi_items_df, n=5):
    """
    結果を処理して保存する関数
    Parameters
    ----------
    models : dict
        訓練済みモデル
        前処理済みのデータ
    sushi_rating_df : pd.DataFrame
        評価データ
    sushi_items_df : pd.DataFrame
        アイテムデータ
    n : int
        予測するアイテム数
    """

    for model_name, model in models.items():
        top_n_recommendations = get_top_n_recommendations(
            model, sushi_rating_df, sushi_items_df, n
        )

        results = []
        for user_id, recommendations in top_n_recommendations.items():
            for rank, (item_id, item_name, predicted_score) in enumerate(
                recommendations, 1
            ):
                results.append(
                    {
                        "user_id": user_id,
                        "rank": rank,
                        "item_id": item_id,
                        "item_name": item_name,
                        "predicted_score": predicted_score,
                    }
                )

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by=["user_id", "rank"])
        df_results.to_csv(
            f"data/{model_name}_results.csv",
            index=False,
            float_format="%.2f",
        )


# メイン処理
score_file_path = "data/sushi_ratings.csv"
items_file_path = "data/sushi_items.csv"
sushi_rating_df = pd.read_csv(score_file_path)
sushi_items_df = pd.read_csv(items_file_path)
models = train_and_predict_models(sushi_rating_df)
process_and_save_results(models, sushi_rating_df, sushi_items_df)
