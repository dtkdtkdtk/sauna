import openai
import os
import streamlit as st
import pandas as pd
import numpy as np
import ast  # ast モジュールをインポート
import json

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )
    st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

file_url = 'https://drive.google.com/uc?id=1FFk8imDrmX22zHsAQjxBG1PxxuW1Uy-T&export=download'
data = pd.read_csv(file_url)
openai.api_key = openai_api_key

system = '''

ユーザーからの入力を以下の#制約に従って変換してください

###制約:
-入力文からサウナ施設名と希望条件を推測してください
-推測したサウナ施設名と希望条件を#出力形式に従って、json形式で出力してください
-#出力例を参考に出力してください

###出力形式:
{"sauna":{ここにサウナ施設名を入力},
 "kibo":{ここに希望条件を入力}}

###出力例
 入力：私が好きなサウナはサウナセンターです。今日はサウナ室の温度が高いサウナに入りたいです。
 回答：
 {"sauna":サウナセンター,
 "kibo":温度の高いサウナ室}

 入力：サウナ北欧に行きたい気分だな。あそこは外気浴が気持ちよいから好き
 回答：
 {"sauna":北欧,
 "kibo":外気浴ができるところ}

 入力：昭和ストロングといえばサンデッキしかかたん
 回答：
 {"sauna":サンデッキ,
 "kibo":昭和ストロング}
 '''

def call_chatgpt(prompt):

  completion = openai.chat.completions.create(
      model='gpt-4o',
      messages = [
    {"role":"system","content":system},
    {"role":"user","content":prompt}
    ])
  return completion.choices[0].message.content


def get_similar_sauna_details(title):
    if title in data['title'].values:
        ranks = data.loc[data['title'] == title, ['rank1', 'rank2', 'rank3', 'rank4', 'rank5']].iloc[0]
    else:
        return "一致するサウナが見つかりませんでした。"

    rank_titles = ranks.tolist()
    result = data[data['title'].isin(rank_titles)]
    return result[['title', 'サマリ', 'url']]

def str_to_float_list(x):
    if isinstance(x, str):
        return [float(item) for item in x.strip('[]').split(',')]
    else:
        return x

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

def call_recomend2(prompt2):

  completion = openai.chat.completions.create(
      model='gpt-4o',
      messages = [
    {"role":"system","content":system3},
    {"role":"user","content":prompt2}
    ])
  return completion.choices[0].message.content



st.title("サウナレコメンダー")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "好きなサウナと希望条件を教えてください"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

   
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    sauna = prompt
    sauna_info = call_chatgpt(sauna)
    st.write("おすすめのサウナを検討中")


    sauna_info = json.loads(sauna_info)
    sauna_name = sauna_info['sauna']

    query = openai.embeddings.create(
        model="text-embedding-3-small",
        input=sauna_name
    )
    input_vector = query.data[0].embedding
    data['サウナベクトル'] = data['サウナベクトル'].apply(str_to_float_list).apply(np.array)

    vector_matrix = np.array(data['サウナベクトル'].tolist())
    similarities = [cosine_similarity(x, input_vector) for x in vector_matrix]

    data['similarity'] = similarities
    most_similar_row = data.loc[data['similarity'].idxmax()]
    input_title = most_similar_row['title']
    similar_sauna_details = get_similar_sauna_details(input_title)
    sauna_list = similar_sauna_details.iloc[:,0].tolist()
    sauna_summary = similar_sauna_details.to_json(orient='records', force_ascii=False)

    sauna_kibo = sauna_info['kibo']
    prompt2 = sauna_kibo

    system3 = f'''

あなたはおすすめのサウナを紹介するスペシャリストです。
ユーザーからの入力に合ったサウナを3つ抽出してください

###制約:
-入力文から希望に合っているサウナを3つ抽出してください。
-サウナは必ず#サウナ施設の中から3つ選択してください。
-選ぶ際には、#サウナ概要の内容を参考に入力文に沿ったもの抽出してください。
-サウナとおすすめの理由を教えてください
-出力はからなず出力形式に従ってください。余計な説明は不要です。

###サウナ施設:
{sauna_list}

###サウナ概要：
{sauna_summary}

###出力形式:
sauna1：おすすめポイント
sauna2：おすすめポイント
sauna3：おすすめポイント


 '''
    reco = call_recomend2(prompt2)
    st.session_state.messages.append({"role": "assistant", "content": reco})
    st.chat_message("assistant").write("あなたにぴったりのおすすめのサウナは以下です。")
    st.chat_message("assistant").write(reco)
