from fastapi import FastAPI
from onnxInference import ColaONNXPredictor
import os
import logging


logger = logging.Logger(__name__)

logger.info("Loading model")
ModelPath = "./models/model.onnx"
predictor = ColaONNXPredictor(ModelPath)
logger.info(f"Model loaded from {ModelPath}")

file_handler = logging.FileHandler("app.log")  # Specify the log file name
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



app = FastAPI(title="MLOPS Basics")

@app.get("/predict")
async def get_prediction(text:str):
    logger.info(f"text recieved:{text}")
    result = predictor.predict(text)

    #Convert any numpy types to native Python types
    result = [{"label": pred["label"], "score": float(pred["score"])} for pred in result]
    return result
    

    