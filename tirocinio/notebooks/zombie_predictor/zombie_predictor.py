import sys
import os
import gzip
import glob
import pandas as pd
import numpy as np
from joblib import load
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, fbeta_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier


COLUMN_NAMES = [ 
    "job_id", "queue", "job_type", "hn",
    "just_started", "just_finished", "js", "nc", 
    "hsj", "hsm", "cpt", "rt", 
    "owner", "ram", "img", 
    "ts", "sn", "disk" 
]
CATEGORICAL_VARS = [*COLUMN_NAMES[1:3], *COLUMN_NAMES[4:6], "owner"]
STRING_VARS = ["job_id", "hn", "sn"]
AGG_COLUMNS = ['ram', 'img', 'disk']
LHC_QUEUES = ['alice', 'atlas', 'cms', 'lhcb']
RESULTS_PATH = 'results.txt'


class InvalidJobs(Exception):
    pass

 
class Preprocessor:
    @staticmethod
    def preprocess(inputf):
        print(f"*** looking for logs at -> {inputf} ***")
        data = pd.concat(map(lambda f: pd.read_csv(f, 
                                                   sep=" ", 
                                                   names=COLUMN_NAMES, 
                                                   dtype={c: "category" for c in CATEGORICAL_VARS}, 
                                                   compression="gzip"), 
                             glob.glob(inputf + "*.gz")))


        data[STRING_VARS] = data[STRING_VARS].astype("string")
        data.insert(0, 'job', data['job_id'] + "_" + data['sn'])
        data['job_work_type'] = data['queue'].str.contains(str.join("|", LHC_QUEUES)).map({True: "lhc", False: "non-lhc"}).astype('category')

        agg_data = data.groupby("job").agg({ 
            'rt': list, 
            'ram': list, 
            'img': list,
            'disk': list, 
            'js': max,
            'job_type': 'first',
            'job_work_type': 'first'
        })

        filtered_agg_data = agg_data[
            (agg_data['rt'].apply(lambda x: x[0] <= 180)) & 
            (agg_data['rt'].apply(lambda x: len(x) >= 20)) &
            (agg_data['js'] == 2)
        ].drop(['rt', 'js'], axis=1).reset_index(drop=False)

        if len(filtered_agg_data) == 0:
            raise InvalidJobs

        for COL in AGG_COLUMNS:
            filtered_agg_data[COL] = filtered_agg_data[COL].apply(
                lambda x: [np.mean(x[i:j]) for i, j in zip([0, 5, 10, 15], [5, 10, 15, 20])]
            )

        return pd.concat([
            filtered_agg_data[filtered_agg_data.columns.difference(AGG_COLUMNS)],
            pd.concat([pd.DataFrame(filtered_agg_data[COL].tolist()).add_prefix(f"{COL}_") for COL in AGG_COLUMNS], axis=1)
        ], axis=1)
    

class Classifier:
    def __init__(self, inputf):
        self.model = load(inputf)

    def predict(self, X):
        print(f"*** getting the predictions for {len(X)} unfinished jobs ***")
        return self.model.predict(X)
        

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python classifier.py <model dump> <log file>")
        exit(1)
                
    try:
        data = Preprocessor.preprocess(args[1])    
    except InvalidJobs:
        print("No jobs valid to predict, try to add more logs!")
    clf = Classifier(args[0])
    np.savetxt(RESULTS_PATH, np.c_[data['job'], clf.predict(data)], fmt=('%s', '%d'))
    print(f"*** saved the predictions to -> '{os.getcwd()}/{RESULTS_PATH}'***")