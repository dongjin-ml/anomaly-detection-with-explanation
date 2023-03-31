import os
import pickle
import argparse
import numpy as np
import pandas as pd
from distutils.dir_util import copy_tree
from sklearn.preprocessing import StandardScaler


class preprocess():
    
    def __init__(self, args):
        
        self.args = args
        self.proc_prefix = self.args.proc_prefix #'/opt/ml/processing'
        
        self.input_dir = os.path.join(self.proc_prefix, "input")
        self.output_dir = os.path.join(self.proc_prefix, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _data_split(self, ):
            
        train_data_ratio = 0.99
        clicks_1T = pd.read_csv(os.path.join(self.input_dir, self.args.train_data_name))
    
        pdTrain = clicks_1T[["page", "user", "click", "residual", "fault"]][:int(clicks_1T.shape[0] * train_data_ratio)]
        pdTest = clicks_1T[["page", "user", "click", "residual", "fault"]][int(clicks_1T.shape[0] * train_data_ratio):]
        print (f"Train: {pdTrain.shape}, Test: {pdTest.shape}")
        
        train_x, train_y = pdTrain[[strCol for strCol in pdTrain.columns if strCol != "fault"]].values, pdTrain["fault"].values.reshape(-1, 1)
        test_x, test_y = pdTest[[strCol for strCol in pdTest.columns if strCol != "fault"]].values, pdTest["fault"].values.reshape(-1, 1)
        print (f'train_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}')
        
        return train_x, train_y, test_x, test_y
    
    def _normalization(self, train_x, test_x):
        
        scaler = StandardScaler()
        scaler.fit(train_x)
        
        train_x_scaled = scaler.transform(train_x)
        test_x_scaled = scaler.transform(test_x)
        
        dump_path = os.path.join(self.output_dir, "StandardScaler", "scaler.pkl")
        os.makedirs(os.path.join(self.output_dir, "StandardScaler"), exist_ok=True)   
        self._to_pickle(scaler, dump_path)
        
        return train_x_scaled, test_x_scaled
    
    def _shingle(self, data):
        
        shingle_size = self.args.shingle_size
        
        num_data, num_features = data.shape[0], data.shape[1]
        shingled_data = np.zeros((num_data-shingle_size+1, shingle_size*num_features))

        print (num_data, shingled_data.shape)

        for idx_feature in range(num_features):

            if idx_feature == 0:
                start, end = 0, shingle_size
            else:
                start = end
                end = start + shingle_size

            for n in range(num_data - shingle_size + 1):
                if n+shingle_size == num_data: shingled_data[n, start:end] = data[n:, idx_feature]    
                else: shingled_data[n, start:end] = data[n:(n+shingle_size), idx_feature]
                
        return shingled_data
    
    def _to_pickle(self, obj, save_path):
        
        with open(file=save_path, mode="wb") as f:
            pickle.dump(obj, f)
    
    def _from_pickle(self, obj_path):
        
        with open(file=obj_path, mode="rb") as f:
            obj=pickle.load(f)
        
        return obj
        

    def execution(self, ):
        
        train_x, train_y, test_x, test_y = self._data_split()
        train_x_scaled, test_x_scaled = self._normalization(train_x, test_x)
        
        train_x_scaled_shingle = self._shingle(train_x_scaled)
        train_y_shingle = self._shingle(train_y)[:, -1].reshape(-1, 1)
        
        test_x_scaled_shingle = self._shingle(test_x_scaled)
        test_y_shingle = self._shingle(test_y)[:, -1].reshape(-1, 1)

        print (f'train_x_scaled_shingle: {train_x_scaled_shingle.shape}')
        print (f'train_y_shingle: {train_y_shingle.shape}')
        print (f'check label: {sum(train_y_shingle == train_y[self.args.shingle_size-1:])}')
        print (f'fault cnt, train_y_shingle: {sum(train_y_shingle)}, train_y: {sum(train_y[self.args.shingle_size-1:])}')
    
        print (f'# features: {test_x_scaled.shape[1]}, shingle_size: {self.args.shingle_size}')
        print (f'test_x_scaled_shingle: {test_x_scaled_shingle.shape}')
        print (f'test_y_shingle: {test_y_shingle.shape}')
        print (f'check label: {sum(test_y_shingle == test_y[self.args.shingle_size-1:])}')
        print (f'fault cnt, train_y_shingle: {sum(test_y_shingle)}, train_y: {sum(test_y[self.args.shingle_size-1:])}')
        
        
        self._to_pickle(train_x_scaled_shingle, os.path.join(self.output_dir, "train_x_scaled_shingle.pkl"))
        self._to_pickle(train_y_shingle, os.path.join(self.output_dir, "train_y_shingle.pkl"))
        self._to_pickle(test_x_scaled_shingle, os.path.join(self.output_dir, "test_x_scaled_shingle.pkl"))
        self._to_pickle(test_y_shingle, os.path.join(self.output_dir, "test_y_shingle.pkl"))
        
        print (self.args.shingle_size, type(self.args.shingle_size))
        print ("data_dir", os.listdir(self.input_dir))
        print ("self.output_dir", os.listdir(self.output_dir))
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_prefix", type=str, default="/opt/ml/processing")
    parser.add_argument("--shingle_size", type=int, default=4)
    parser.add_argument("--train_data_name", type=str, default="merged_clicks_1T.csv")
    
    
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    prep = preprocess(args)
    prep.execution()