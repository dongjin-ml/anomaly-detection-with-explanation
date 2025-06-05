import os
import json
import torch
import pickle
import mlflow
import tarfile
import argparse
import torch.nn as nn

import numpy as np
from autoencoder import AutoEncoder, get_model

NUM_FEATURES, SHINGLE_SIZE, EMB_SIZE = 4, 4, 4
FEATURE_NAME = ["URLS", "USERS", "CLICKS", "RESIDUALS"]

class evaluation():
    
    def __init__(self, args):
        
        self.args = args
        self.proc_prefix = self.args.proc_prefix #'/opt/ml/processing'

        self.input_dir = os.path.join(self.proc_prefix, "test")
        self.output_dir = os.path.join(self.proc_prefix, "output")
        self.model_dir = os.path.join(self.proc_prefix, "model")

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self._extract_model_artifacts()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.mlflow_tracking_arn = self.args.mlflow_tracking_arn
        self.experiment_name = self.args.experiment_name
        self.mlflow_run_name = self.args.mlflow_run_name
        
        print ("MLFLOW_TRACKING_ARN", self.mlflow_tracking_arn)
        print ("experiment_name", self.experiment_name)
        print ("run_name", self.mlflow_run_name)
        
    def _extract_model_artifacts(self, ):
        """모델 아티팩트 압축 해제"""
        tar_file_path = os.path.join(self.model_dir, 'model.tar.gz')

        if os.path.exists(tar_file_path):
            with tarfile.open(tar_file_path, 'r:gz') as tar:
                tar.extractall(path=self.model_dir)
            print(f"모델 아티팩트가 {self.model_dir}에 압축 해제되었습니다.")
        else:
            print(f"압축 파일을 찾을 수 없습니다: {tar_file_path}")
            
    def _load_model(self, ):
    
        input_dim = NUM_FEATURES*SHINGLE_SIZE + EMB_SIZE
        model = get_model(
            input_dim=input_dim,
            hidden_sizes=[64, 48],
            btl_size=32,
            emb_size=EMB_SIZE
        )
        print (f'Input dim: {input_dim}, from num_features({NUM_FEATURES}), shingle_size({SHINGLE_SIZE}) and emb_size({EMB_SIZE})')

        with open(os.path.join(self.model_dir, "best_model.pth"), "rb") as f:
            model.load_state_dict(torch.load(f))
        return model.to(self.device).eval()
    
    def _from_pickle(self, obj_path):
        with open(file=obj_path, mode="rb") as f:
            obj=pickle.load(f)
        return obj
    
    def _prediction(self, model, data):
        
        anomaly_calculator = nn.L1Loss(reduction="none").to(self.device)
                
        pred_results = []
        for idx, record in enumerate(data):

            if idx % 1000 == 0: print (f'{idx}/{data.shape[0]}')
            
            input_data = record.tolist()
            x = torch.tensor(input_data, dtype=torch.float32)
            x = x.unsqueeze(0).to(self.device)
            
            anomal_scores = []
            with torch.no_grad():

                time, x = x[:, 0].type(torch.int), x[:, 1:]

                t_emb, _x = model.forward(time, x)
                x = torch.cat([t_emb, x], dim=1)


                anomal_score = anomaly_calculator(x[:, EMB_SIZE:], _x[:, EMB_SIZE:]) # without time
                anomal_score_sap = 0
                for layer in model.encoder.layer_list:
                    x, _x = layer(x), layer(_x)
                    diffs = anomaly_calculator(x, _x)
                    anomal_score_sap += (diffs).mean(dim=1)

                for record, sap in zip(anomal_score.cpu().numpy(), anomal_score_sap.cpu().numpy()):
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
                    
            pred_results.append(anomal_scores)
        
        return pred_results
    
    def _metric(self, pred, real):
        
        mlflow.set_tracking_uri(self.mlflow_tracking_arn)
        mlflow.set_experiment(self.experiment_name)
        
        filter_string = f"run_name='{self.mlflow_run_name}'"
        run_id = mlflow.search_runs(filter_string=filter_string)["run_id"][0]
        print ("filter_string", filter_string)
        print ("mlflow.search_runs(filter_string=filter_string)", mlflow.search_runs(filter_string=filter_string))
        print ("run_id", run_id)
        params = {k: o for k, o in vars(self.args).items()}
        
        
        
        with mlflow.start_run(run_id=run_id, log_system_metrics=True):
            with mlflow.start_run(run_name="Evaluation", log_system_metrics=True, nested=True) as evaluation_run:
                
                ##############################################################################################
                #
                #    your logic to get performance metric such as precision, recall, f-sccore, AUROC AUPR etc.
                #
                ##############################################################################################
                
                
                mlflow.log_params({**params})
                mlflow.autolog()
                
                accuracy = 0.8
                precision = 0.75
                recall = 0.85
                f1 = 0.88

                report_dict = {
                    "metrics": {
                        "accuracy": {
                            "value": accuracy,
                            "standard_deviation": None
                        },
                        "precision": {
                            "value": precision,
                            "standard_deviation": None
                        },
                        "recall": {
                            "value": recall,
                            "standard_deviation": None
                        },
                        "f1": {
                            "value": f1,
                            "standard_deviation": None
                        },
                    },
                }
                
                mlflow.log_metric(
                    key='accuracy',
                    value=accuracy,
                    step=0
                )

                mlflow.log_metric(
                    key='precision',
                    value=precision,
                    step=0
                )

                mlflow.log_metric(
                    key='recall',
                    value=recall,
                    step=0
                )
                
                mlflow.log_metric(
                    key='f1',
                    value=f1,
                    step=0
                )
        
        return report_dict
         
    def execution(self, ):
        
        # load model
        model = self._load_model()
        
        # pred
        data_path = os.path.join(self.input_dir, "data_x_scaled_shingle.pkl")
        data = self._from_pickle(data_path)
        pred_results = self._prediction(model, data)
        
        # metric
        real_y = os.path.join(self.input_dir, "data_y_shingle.pkl")
        report_dict = self._metric(pred_results, real_y)
        
        # save metric
        evaluation_path = f"{self.output_dir}/evaluation.json"
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))
        
        
        print ("self.input_dir", os.listdir(self.input_dir))
        print ("==")
        print ("self.output_dir", os.listdir(self.output_dir))
        print ("==")
        print ("self.model_dir", os.listdir(self.model_dir))
        print ("==")

        print ("data_dir", os.listdir(self.input_dir))
        print ("self.output_dir", os.listdir(self.output_dir))
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_prefix", type=str, default="/opt/ml/processing")
    parser.add_argument("--mlflow_tracking_arn", type=str, default="mlflow_tracking_arn")
    parser.add_argument("--experiment_name", type=str, default="experiment_name")
    parser.add_argument("--mlflow_run_name", type=str, default="mlflow_run_name")

    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
        
    evaluator = evaluation(args)
    evaluator.execution()