# -*- coding: utf-8 -*-
"""
Created on Wed Apr 5 19:12:48 2023

@author: Neal
"""
from pydantic import BaseModel
import model.sentiment_model as clf
from fastapi import FastAPI
from joblib import load
import uvicorn
        
app = FastAPI(title="MDS5724 Group Project - Task2 - Demo", 
              description="API for Text Sentiment Analysis", version="1.0")

class Payload(BaseModel):
    news_title: str = ""

@app.on_event('startup')
def load_model():
    clf.model = load('model/text_sentiment_model_v001.joblib')


@app.post('/predict')
async def get_prediction(payload: Payload = None):
    news_title = dict(payload)['news_title']
    score = clf.model.predict([news_title,]).tolist()[0]
    return score

if __name__ == '__main__':
    uvicorn.run(app, port=5724, host='0.0.0.0')