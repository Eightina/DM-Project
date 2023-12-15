# -*- coding: utf-8 -*-
"""
Created on Wed Apr 5 19:12:48 2023

@author: Neal
"""
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, TextClassificationPipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

app = FastAPI(title="MDS5724 Group Project - Task2 - Demo", 
              description="API for Text Sentiment Analysis", version="1.0")

class Payload(BaseModel):
    news_title: str = "Nice Day High Profit"
    
class LocalModel:
    pass
    
def pipe_predict(pipe, x):
    pred_res = pipe(x)
    if pred_res[0][0]['score'] > pred_res[0][1]['score']:
        pred_res = pred_res[0][0]
    else:
        pred_res = pred_res[0][1]
    if pred_res['label'] == 'LABEL_0':
        pred_res['label'] = -1
    else:
        pred_res['label'] = 1
    final_res = "{{“label”:{}, “sentiment”:{}}}".format(
            pred_res['label'],
            pred_res['score']
        )
    return final_res

@app.on_event('startup')
def load_model():
    LocalModel.tokenizer = AutoTokenizer.from_pretrained("./model/test-trainer/onnx_disroberta")
    LocalModel.model = ORTModelForSequenceClassification.from_pretrained("./model/test-trainer/onnx_disroberta", )


@app.post('/predict')
async def get_prediction(payload: Payload = None):
    news_title = dict(payload)['news_title']
    if len(news_title) > 512:
        news_title = news_title[:512]
    # print(news_title)
    # return news_title
    pipe = TextClassificationPipeline(model=LocalModel.model, tokenizer=LocalModel.tokenizer, return_all_scores=True)
    return pipe_predict(pipe, news_title)

if __name__ == '__main__':
    uvicorn.run(app, port=5724, host='0.0.0.0')