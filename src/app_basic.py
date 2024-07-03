import streamlit as st
import pandas as pd
import os
import glob

# サイドバーの設定
st.sidebar.title("設定")

# CSVファイルのリストを取得
csv_files = glob.glob(os.path.join("recommend_score", "*.csv"))

# ファイル選択のプルダウン
selected_file = st.sidebar.selectbox("出力ファイルを選択してください", csv_files)

# メインの処理
if selected_file:
    # データの読み込み
    sushi_ratings = pd.read_csv(os.path.join("preprocessed_data", "sushi_ratings.csv"))
    output_data = pd.read_csv(selected_file)

    # ユーザーIDの選択
    user_ids = sorted(sushi_ratings["user_id"].unique())
    selected_user_id = st.sidebar.select_slider("ユーザーIDを選択", options=user_ids)

    # 選択されたユーザーのデータを取得
    user_ratings = sushi_ratings[
        sushi_ratings["user_id"] == selected_user_id
    ].sort_values("score", ascending=False)
    user_predictions = output_data[
        output_data["user_id"] == selected_user_id
    ].sort_values("predicted_rating", ascending=False)

    # 実際の評価の表示（2行5列）
    st.header("実際の評価値")
    for row in range(2):
        cols = st.columns(5)
        for col in range(5):
            index = row * 5 + col
            if index < len(user_ratings) and index < 10:
                item = user_ratings.iloc[index]
                cols[col].write(f"{item['name']}  \nスコア: {item['score']}")

    # 予測評価の表示（1行5列）
    st.header("予測評価値")
    cols = st.columns(5)
    for col in range(5):
        if col < len(user_predictions):
            item = user_predictions.iloc[col]
            cols[col].write(
                f"{item['item_name']}  \n予測スコア: {item['predicted_rating']:.2f}"
            )
else:
    st.write("CSVファイルを選択してください。")
