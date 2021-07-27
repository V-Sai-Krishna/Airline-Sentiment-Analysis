from util import *
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from preprocessor import ProcessText
from feature import extract_features
app = FastAPI()

@app.get("/")
async def create_item(query: str):
    model= pickle.load(open('enthirebestmodel.sav', 'rb'))
    ptf=ProcessText()
    x1,x2=ptf.process([query])
    xtrain=extract_features(x2[0])
    y_pred = model.predict(xtrain.reshape(1,-1))
    if y_pred[0]==0:
        return {"That is a negative statement"}
    else:
        return {"That is a positive statement"}