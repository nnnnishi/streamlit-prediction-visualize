import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image

# クエリパラメータの取得
query_params = st.query_params

# サイドバーの設定
st.sidebar.title("設定")

# CSVファイルのリストを取得
csv_files = glob.glob(os.path.join("recommend_score", "*.csv"))

# モデルの選択（クエリパラメータがある場合はそれを使用）
default_model_a = query_params.get("model_a", csv_files[0] if csv_files else None)
default_model_b = query_params.get(
    "model_b", csv_files[1] if len(csv_files) > 1 else None
)

selected_file_a = st.sidebar.selectbox(
    "モデルAの出力ファイルを選択してください",
    csv_files,
    index=csv_files.index(default_model_a) if default_model_a in csv_files else 0,
)
selected_file_b = st.sidebar.selectbox(
    "モデルBの出力ファイルを選択してください",
    csv_files,
    index=csv_files.index(default_model_b)
    if default_model_b in csv_files
    else 1
    if len(csv_files) > 1
    else 0,
)

# ローマ字のitem_nameから日本語のitem_name_jaに変換する辞書を作成
item_names = pd.read_csv(os.path.join("data", "name_mapping.csv"))
en2ja = dict(zip(item_names["name"], item_names["name_ja"]))

# メインの処理
if selected_file_a and selected_file_b:
    # データの読み込み
    sushi_ratings = pd.read_csv(os.path.join("data", "sushi_ratings.csv"))
    output_data_a = pd.read_csv(selected_file_a)
    output_data_b = pd.read_csv(selected_file_b)

    # ユーザーIDの選択（クエリパラメータがある場合はそれを使用）
    user_ids = sorted(sushi_ratings["user_id"].unique())
    default_user_id = query_params.get("user_id", user_ids[0])
    selected_user_id = st.sidebar.select_slider(
        "ユーザーIDを選択",
        options=user_ids,
        value=int(default_user_id) if str(default_user_id).isdigit() else user_ids[0],
    )

    # クエリパラメータの更新
    st.query_params["model_a"] = os.path.basename(selected_file_a)
    st.query_params["model_b"] = os.path.basename(selected_file_b)
    st.query_params["user_id"] = str(selected_user_id)

    # 予測値スコアの上限と下限を入力するフォーム
    st.sidebar.subheader("予測値スコアの範囲")
    score_min = st.sidebar.number_input(
        "下限", min_value=0.0, max_value=10.0, value=0.0, step=0.1
    )
    score_max = st.sidebar.number_input(
        "上限", min_value=0.0, max_value=10.0, value=10.0, step=0.1
    )

    # 選択されたユーザーのデータを取得
    user_ratings = sushi_ratings[
        sushi_ratings["user_id"] == selected_user_id
    ].sort_values("score", ascending=False)

    user_predictions_a = output_data_a[
        (output_data_a["user_id"] == selected_user_id)
        & (output_data_a["predicted_rating"] >= score_min)
        & (output_data_a["predicted_rating"] <= score_max)
    ].sort_values("predicted_rating", ascending=False)

    user_predictions_b = output_data_b[
        (output_data_b["user_id"] == selected_user_id)
        & (output_data_b["predicted_rating"] >= score_min)
        & (output_data_b["predicted_rating"] <= score_max)
    ].sort_values("predicted_rating", ascending=False)

    # 実際の評価の表示（2行5列）
    st.header("実際の評価値")
    for row in range(2):
        cols = st.columns(5)
        for col in range(5):
            index = row * 5 + col
            if index < len(user_ratings) and index < 10:
                item = user_ratings.iloc[index]
                image_path = os.path.join("images", f"{item['name']}.png")
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    cols[col].image(
                        image,
                        caption=f'{en2ja[item["name"]]} {item["score"]}',
                        use_column_width=True,
                    )

    # モデルAの予測評価の表示（1行5列）
    st.header(f"モデルAの予測評価値 ({os.path.basename(selected_file_a)})")
    cols = st.columns(5)
    for col in range(5):
        if col < len(user_predictions_a):
            item = user_predictions_a.iloc[col]
            image_path = os.path.join("images", f"{item['item_name']}.png")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                cols[col].image(
                    image,
                    caption=f'{en2ja[item["item_name"]]} {item["predicted_rating"]:.2f}',
                    use_column_width=True,
                )

    # モデルBの予測評価の表示（1行5列）
    st.header(f"モデルBの予測評価値 ({os.path.basename(selected_file_b)})")
    cols = st.columns(5)
    for col in range(5):
        if col < len(user_predictions_b):
            item = user_predictions_b.iloc[col]
            image_path = os.path.join("images", f"{item['item_name']}.png")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                cols[col].image(
                    image,
                    caption=f'{en2ja[item["item_name"]]} {item["predicted_rating"]:.2f}',
                    use_column_width=True,
                )

else:
    st.write("両方のCSVファイルを選択してください。")
