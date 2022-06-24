from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import dialogflow
from pydantic import BaseModel
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from konlpy.tag import Mecab
from tqdm import tqdm
import requests


PROJECT_ID = "lawcat-9hbr"
SESSION_ID = "123456789"
LANGUAGE_CODE = "ko-KR"


chat_history = {}

cache = []

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/lawcat_ai/key/lawcat-9hbr-71201f0f69e1.json"

model = SentenceTransformer('paraphrase-distilroberta-base-v1').cuda()
df = pd.read_csv("data/판례.csv").fillna('')
judgement_df = pd.read_csv("data/judgementNote.csv")

m = Mecab()

term_df = pd.read_csv("data/용어사전.csv")

embedding_dict = {}

def convert_nng(x):
    return ' '.join([word[0] for word in m.pos(x) if word[1] == "NNG"])

judgement_df['judgementNote'] = judgement_df['judgementNote'].apply(convert_nng)
case_embd = model.encode(judgement_df['judgementNote'], convert_to_tensor=True).cuda()

for i, row in tqdm(judgement_df.iterrows(), total=len(judgement_df)):
    # if i==20000:
        # break
    # embedding_dict[row['caseNumber']] = model.encode(row['judgementNote'], convert_to_tensor=True).cuda()
    embedding_dict[row['caseNumber']] = case_embd[i]


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    contents: str
    channelId: str
    hash: str

@app.post("/intent")
def send_intent(body: Item):
    if body.hash in cache:
        return

    cache.append(body.hash)

    print(body)
    text = body.contents
    channelId = body.channelId

    answer_list = []

    result = detect_intent_texts(PROJECT_ID, SESSION_ID, [text], LANGUAGE_CODE)

    intent = result.intent.display_name
    answer = result.fulfillment_text

    keyword = m.pos(text)[0][0]

    if channelId not in chat_history.keys():
        chat_history[channelId] = [keyword]
    else:
        chat_history[channelId].append(keyword)

    if intent == "용어":
        answer = answer.replace("#", keyword)

    if intent == "판례" or intent == "용어":
        if channelId not in chat_history.keys():
            chat_history[channelId] = [keyword]
        else:
            chat_history[channelId].append(keyword)
        answer_list = post_inference(intent, keyword, text, channelId)
    elif intent == "변호사":
        if len(chat_history[channelId]) == 0:
            answer = "먼저 겪은 사례를 말해줘! 잘 맞는 변호사를 추천해줄게!"
            answer_list.append({
                "isLawyer": True,
                "keyword": keyword,
                "answer": answer
            })
        else:
            answer_list.append({
                "isLawyer": True,
                "keyword": chat_history[channelId][-1],
                "answer": answer
            })
    else:
        answer_list.append({
            "isLawyer": False,
            "keyword": keyword,
            "answer": answer
        })

    # await post_inference(intent, keyword, text, channelId)

    # answer_list.append({
    #     "isLawyer": isLawyer,
    #     "keyword": keyword,
    #     "answer": answer
    # })

    return {
        "content": answer_list,
    }
    

def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    print("Session path: {}\n".format(session))

    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)

        query_input = dialogflow.QueryInput(text=text_input)

        response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )

        print("=" * 20)
        print("Query text: {}".format(response.query_result.query_text))
        print(
            "Detected intent: {} (confidence: {})\n".format(
                response.query_result.intent.display_name,
                response.query_result.intent_detection_confidence,
            )
        )
        print("Fulfillment text: {}\n".format(response.query_result.fulfillment_text))

    return response.query_result


def post_inference(intent, keyword, text, channelId):
    answer_list = []

    if intent == "판례":
        score_dict = {}

        text = ' '.join([word[0] for word in m.pos(text) if word[1] == "NNG"]).replace("판례", "")
        query_embd = model.encode(text, convert_to_tensor=True).cuda()
        for key in embedding_dict.keys():
            score_dict[key] = util.pytorch_cos_sim(query_embd, embedding_dict[key]).item()

        max_caseNumber = max(score_dict, key=score_dict.get)
        print(score_dict[max_caseNumber])

        item = df[df['caseNumber'] == max_caseNumber]
        print(item)

        caseName = item['caseName'].tolist()[0]
        sentenceDate = item['sentenceDate'].tolist()[0]
        # judgementAbstract = item['judgementAbstract']
        judgementNote = item['judgementNote'].tolist()[0]
        refArticle = item['refArticle'].tolist()[0]

        output = f"판례명: {str(caseName)}\n\n선고일: {sentenceDate}\n\n판례 내용: {str(judgementNote)}\n\n관련 형법: {refArticle}"
        print(output)

        answer_list.append({
            "isLawyer": False,
            "keyword": keyword,
            "answer": output
        })
        answer_list.append({
            "isLawyer": False,
            "keyword": keyword,
            "answer": "더 궁금한 거 있어?"
        })

        # url = "http://118.67.130.115/api/complete"
        # data = {
        #     "content": answer_list
        # }
        # response = requests.post(url, data=data)
    elif intent == "용어":
        print(keyword)
        if keyword in list(term_df['용어']):
            answer = f"{term_df[term_df['용어'] == keyword]['용어설명'].tolist()[0]}?"
            answer_list.append({
                "isLawyer": False,
                "keyword": keyword,
                "answer": answer.rstrip("?")
            })
            answer_list.append({
                "isLawyer": False,
                "keyword": keyword,
                "answer": "어때, 나 잘 찾지?"
            })
        else:
            answer = "미안, 내가 아는 용어가 아니네!"
            answer_list.append({
                "isLawyer": False,
                "keyword": keyword,
                "answer": answer.rstrip("?")
            })

        print(answer)

        answer_list.append({
            "isLawyer": False,
            "keyword": keyword,
            "answer": "더 궁금한 거 있어?"
        })

    return answer_list

        # url = "http://118.67.130.115/api/complete"
        # data = {
        #     "content": answer_list,
        #     "channelId": channelId,
        # }
        # response = requests.post(url, data=data)