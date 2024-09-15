# DSの結果をアプリで確認

## データセット
以下のSUSHI Preference Data Setsを利用  
https://www.kamishima.net/sushi/

## 処理方法
- sushi3-2016をrawdataディレクトリ以下に配置
- src/preprocess_rawdata.pyを利用してデータの前処理
- src/recommend_basic.pyを利用して協調フィルタリングによるスコアリング
- streamlit run work/app_basic.pyでアプリケーション起動
