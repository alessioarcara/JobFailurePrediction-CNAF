import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import scipy.stats as st
from sklearn.model_selection import cross_validate, KFold, RandomizedSearchCV
from util import query
random_state = 42

# librerie grafiche
from matplotlib import pyplot as plt
import seaborn as sns

LABELS = ["short", "medium", "long"]

# import nltk
# STOPWORDS = nltk.corpus.stopwords.words('english')

def load_data(path, start_date, end_date, min_runtime, conn_string="postgresql://accguy:accguy@192.168.1.17/htmnew"):
    if os.path.exists(path):
        print("CACHE")
        df = pd.read_parquet(path)
    else:
        print("DOWNLOAD")
        df = pd.read_sql(query.jobs_from_date_to_date, conn_string, params=((start_date, min_runtime, end_date, min_runtime, start_date, end_date, min_runtime)))
        print("SAVING")
        df.to_parquet(path, compression='snappy')
    return df

def plot_multiple_subplots(data, mainPlotCallback, nrows, ncols, figsize=None, plotTitle=None, remainingPlotsCallbacks=[]):
    """
    Parameters
    ----------
    data : pandas DataFrame
    mainPlotCallback : main plot function to draw multiple plots. 
    remainingPlotsCallbacks : list of callbacks that are used to draw on remaining plots.
    """
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    flattened_axs = axs.ravel()
    plt.subplots_adjust(hspace=0.45)
    
    c = 0
    for key in data:
        mainPlotCallback(data, key, flattened_axs[c])
        c += 1
    for callback in remainingPlotsCallbacks:
        callback(flattened_axs[c])
        c += 1
        
    plt.suptitle(plotTitle, fontsize=16, y=1)
    plt.tight_layout()
    plt.show()
    
def bin_df(df, lower_bound, upper_bound):
    return df.groupby(pd.cut(df.index, bins = [0, lower_bound, upper_bound, len(df)], right=False)).sum()

# def preprocess_text(raw_text , filters=None , stopwords=STOPWORDS):
#     """
#         * filter punctuation
#         * to_lower
#         * remove stop words (from nltk corpus)
#         * remove multiple spaces with one
#         * remove leading spaces    
#     """
#     # filter punctuation and case conversion
#     translation_table = {ord(char): ord(' ') for char in filters}
#     raw_text = raw_text.str.translate(translation_table)
#     raw_text = raw_text.str.lower()
#     # remove words whose word length is less than 2 characters
#     raw_text = raw_text.str.findall('\w{4,}').str.join(' ')
#     # remove stopwords
#     stopwords_gpu = cudf.Series(stopwords)
#     raw_text = raw_text.str.replace_tokens(stopwords_gpu, ' ')
#     # replace multiple spaces with single one and strip leading/trailing spaces
#     raw_text = raw_text.str.normalize_spaces( )
#     raw_text = raw_text.str.strip(' ')
#     return raw_text

def bin_job_runtime(vect_runtime: pd.Series, lower_bound = 6, upper_bound = 30):
    return pd.cut(vect_runtime / 3600.0, bins = [-float("inf"), lower_bound, upper_bound, len(vect_runtime)], right=False, labels=LABELS)

def nested_cross_validation(X, y, model_to_tune, space = dict()):
    # configure the cross-validation procedure
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # nested CV with parameter optimization
    search = RandomizedSearchCV(estimator=model_to_tune, param_distributions=space, scoring=scorer, cv=inner_cv)
    result = cross_validate(search, X, y, cv=outer_cv, scoring=scorer, return_estimator=True, n_jobs=-1, verbose=2)
    
    scores = result['test_score']  # Equivalent to output of cross_val_score()
    best_models = result['estimator']
    
    for score, model in zip(scores, best_models):
        print('>est=%.3f, cfg=%s' % (score, model.best_params_))
    # stima delle performance del modello a regime sui
    print('f1_score: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

def confidence_interval(N, acc, alpha=0.05, verbose=False):
    if verbose:
        print(f"\n *** Calcolo intervallo di confidenza per alpha: {alpha}, N: {N} ***\n")
            
    Z = st.norm.ppf(1-alpha/2)
    denom = 2*(N+Z**2)
    p_min = (2 * N * acc + Z**2 - Z * (Z**2 + 4 * N * acc -4 * N * acc**2)**.5)/denom
    p_max = (2 * N * acc + Z**2 + Z * (Z**2 + 4 * N * acc -4 * N * acc**2)**.5)/denom
    
    return p_min, p_max

def eval_model(X, y, model, threshold=0.5, alpha=0.05, labels=[], verbose=False):
    y_pred = model.predict(X)
    if threshold is not None:
        y_pred = (y_pred >= threshold).astype(int)
        # y_pred = (np.sum(np.square(y_pred - X), axis=1) > thr).astype(int)
    
    metrics = ["precision", "recall", "f1_measure"]
        
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average=None)
    recall = recall_score(y, y_pred, average=None)
    f1_measure = f1_score(y, y_pred, average=None)
    all_classes = pd.Series([precision.mean(), recall.mean(), f1_measure.mean()],  index=metrics)
        
    if verbose:
        print("\n*** Confusion matrix ***\n")
        cf_matrix = confusion_matrix(y, y_pred)
        sns.heatmap(cf_matrix, annot=True, cmap = "Blues", fmt="d", xticklabels=labels, yticklabels=labels)
        plt.title('Confusion matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
        print("\n*** Precision, Recall, F1-measure per classe e media ***\n")
        model_stats = pd.concat(
            [pd.DataFrame([precision, recall, f1_measure], index=metrics), all_classes], axis=1)
        model_stats.columns = labels + ["all"]
        print(model_stats)

        print(f"\n*** Calcolo intervallo di confidenza con Confidenza={1-alpha} con N={X.shape[0]} per accuracy e f1-measure ***\n")
        print(f"accuracy: ({accuracy}), intervallo confidenza: {confidence_interval(X.shape[0], accuracy, alpha)}")
        print(f"f1-measure: ({f1_measure.mean()}), intervallo confidenza: {confidence_interval(X.shape[0], f1_measure.mean(), alpha)}")

    return (accuracy, f1_measure.mean())

def eval_difference_two_model(acc1, acc2, N1, N2, alpha=0.05, confidence_level=False):
    print(f"\n*** Valutazione statistica differenza tra modello 1 e modello 2 ***")
    print(f"(acc: {acc1}, N: {N1}) (acc: {acc2}, N: {N2})\n")
    Z = st.norm.ppf(1-alpha/2)
    e1 = 1 - acc1; e2 = 1 - acc2
    d= abs(e2-e1)
    var_d = (e1*(1-e1))/N1 + (e2*(1-e2))/N2
    d_min = d - Z * var_d**0.5
    d_max = d + Z * var_d**0.5
    
    if confidence_level:
        print(f"\n*** Valutazione soglia confidenza che rende significativa la differenza tra i due modelli ***")
        print(f"a: {round(st.norm.sf(d/var_d**0.5) * 2, 2)}\n")
    
    return d_min, d_max