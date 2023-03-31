import os
import sys
strBasePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(strBasePath)

import io
import json
import torch
import pickle
import logging
import traceback
import numpy as np
import pandas as pd
import torch.nn as nn
from autoencoder import AutoEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NUM_FEATURES, SHINGLE_SIZE = 4, 4
FEATURE_NAME = ["URLS", "USERS", "CLICKS", "RESIDUALS"]

class json_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
    
def model_fn(model_dir):
    
    logger.info("### model_fn ###")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_dim = NUM_FEATURES*SHINGLE_SIZE
    model = AutoEncoder(input_dim=input_dim)
    logger.info(f'Input dim: {input_dim}, from num_features({NUM_FEATURES}) and shingle_size({SHINGLE_SIZE})')
    print (f'Input dim: {input_dim}, from num_features({NUM_FEATURES}) and shingle_size({SHINGLE_SIZE})')
    
    with open(os.path.join(model_dir, "best_model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device).eval()

def input_fn(request_body, request_content_type):
  
    logger.info("### input_fn ###")
    logger.info(f"content_type: {request_content_type}")   
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if isinstance(request_body, str): ## json
        print ("request_body: string")
    elif isinstance(request_body, io.BytesIO):
        print ("request_body: io.BytesIO")
        request_body = request_body.read()
        request_body = bytes.decode(request_body)        
    elif isinstance(request_body, bytes):
        print ("request_body: bytes")
        request_body = request_body.decode()
        
    try:
        if request_content_type=='application/json':
            deserialized_input = json.loads(request_body)
            input_data = deserialized_input["INPUT"]
            input_dtype = deserialized_input["DTYPE"]
            if input_dtype in ["float32", "float64"]: dtype=torch.float32
        
        elif request_content_type=='application/pickle':
            deserialized_input = pickle.loads(request_body)
            input_data = deserialized_input["INPUT"]
            input_dtype = deserialized_input["DTYPE"]
            if input_dtype in ["float32", "float64"]: dtype=torch.float32
        
        else:
            ValueError("Content type {} is not supported.".format(content_type))

        input_data = torch.tensor(input_data, dtype=dtype)
        input_data = input_data.unsqueeze(0)
             
    except Exception:
        print(traceback.format_exc())  

    return input_data.to(device)

def predict_fn(x, model):

    model.eval()
    
    logger.info("### predict_fn ###")    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    anomaly_calculator = nn.L1Loss(reduction="none").to(device)
    
    try:
        logger.info(f"#### type of input data: {type(x)}")                                  
        
        anomal_scores = []
        with torch.no_grad():
            _x = model.forward(x)
            e1, e2 = model.forward(x, hidden=True)
            _e1, _e2 = model.forward(_x, hidden=True)
            
            anomal_score = anomaly_calculator(x, _x)
            sap_score = anomaly_calculator(x, _x).mean(dim=1) + anomaly_calculator(e1, _e1).mean(dim=1) + anomaly_calculator(e2, _e2).mean(dim=1)
            
            for record, sap in zip(anomal_score.cpu().numpy(), sap_score.cpu().numpy()):
                dicScore = {"ANOMALY_SCORE_SAP": sap}
                for cnt, idx in enumerate(range(0, SHINGLE_SIZE*NUM_FEATURES, SHINGLE_SIZE)):
                    start = idx
                    end = start + SHINGLE_SIZE
                    dicScore[FEATURE_NAME[cnt] + "_ATTRIBUTION_SCORE"] = np.mean(record[start:end])

                total_socre = 0
                for k, v in dicScore.items():
                    if k not in ["fault", "ANOMALY_SCORE_SAP"]: total_socre += v
                dicScore["ANOMALY_SCORE"] = total_socre
                anomal_scores.append(dicScore)
                
        logger.info(f'predictions: {anomal_scores}')    
        
    except Exception:
        print(traceback.format_exc())        
        
    return anomal_scores

def output_fn(predictions, content_type="application/json"):
    
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    
    logger.info("### output_fn ###") 
    
    if content_type == "application/json":
        outputs = json.dumps(
            {'pred': predictions},
            cls=json_encoder
        )             
        return outputs
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))

