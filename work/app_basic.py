# 実行はstreamlit-prediction-visualizeの階層で streamlit run work/app_basic.pyとする
import streamlit as st
import pandas as pd
import os
import glob

# サイドバーの設定
st.sidebar.title("設定")

# CSVファイルのリストを取得
csv_files = glob.glob(os.path.join("data", "*_results.csv"))

# ファイル選択のセレクトボックス
selected_file = st.sidebar.selectbox("出力ファイルを選択してください", csv_files)

# メインの処理
if selected_file:
    st.header("選択されたファイル")
    st.write(selected_file)
else:
    st.write("CSVファイルを選択してください。")
