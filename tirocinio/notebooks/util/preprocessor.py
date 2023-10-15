import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


TIME_SERIES_COLS = ['ram', 'swap', 'disk']
TIME_STEP_COL = 't'
STRING_COLS = ['job', 'queue']
TARGET_COL = 'too_much_time'
CAT_COLS = ['job_work_type', 'job_type', 'too_much_time']


class Preprocessor:
    def __init__(self, random_state):
        self.random_state = random_state
        self.column_transformer = ColumnTransformer([
            ('num', StandardScaler(), TIME_SERIES_COLS),
            ('cat', OneHotEncoder(handle_unknown="ignore"), CAT_COLS[:2]),
        ], remainder="drop") 
        
    def preprocess(self, data, window_size=5, perc_undersample=1, split=False, format_type="rows"):
        self.__define_job_work_and_type(data)
        self.__calculate_days_and_labels(data)
        self.__remove_duplicated_jobs(data)
        self.__remove_missing_values(data)
        self.__remove_jobs_shorter_than_one_hour(data)
        data = self.__random_undersample(data, perc_undersample)
        self.__downsample_data(data, window_size)
        if split == True:
            print("--- Splitting data in train and val data ---")
            train, val = train_test_split(data, test_size=0.2, stratify=data[TARGET_COL], random_state=self.random_state)
            return self.__transform_arrays_to_format(train, format_type), self.__transform_arrays_to_format(val, format_type)
        return self.__transform_arrays_to_format(data, format_type)
    
    def transform(self, data, fit: bool = False) -> (np.array, np.array):
        y = data.pop(TARGET_COL)[::96].reset_index(drop=True)
        if fit:
            transformed_data = self.column_transformer.fit_transform(data)
        else:
            transformed_data = self.column_transformer.transform(data)
        return self.__transform_to_tensor(transformed_data), y
        
    def __downsample(self, x, w):
        padded_x = np.pad(x, (0, w - len(x) % w), mode='edge') if len(x) % w != 0 else x
        return np.mean(padded_x.reshape(-1, w), axis=1)
             
    def __downsample_data(self, data, w):
        print(f"--- Downsampling time series from 3m to {w*3}m ---")
        for COL in TIME_SERIES_COLS + [TIME_STEP_COL]:
            data[COL] = data[COL].apply(lambda x: self.__downsample(x, w))
        
    def __define_job_work_and_type(self, data):
        print("--- Defining 'job_work_type' and 'job_type' columns ---")
        data[STRING_COLS] = data[STRING_COLS].astype(str)
        search_for_queue = '|'.join(['alice', 'atlas', 'cms', 'lhcb'])
        data['job_work_type'] = data['queue'].str.contains(search_for_queue).map({True: "lhc", False: "non-lhc"})
        data['job_type'] = data['job'].str.contains('ce').map({True: "grid", False: "local"})

    def __calculate_days_and_labels(self, data):
        print("--- Calculating 'days' and 'labels' columns ---")
        def set_too_much_time(row):
            days_limit = 3 if row['job_type'] == 'grid' else 6
            return 1 if row['days'] > days_limit and row['fail'] == 1 else 0
        
        labels = np.arange(1, 8)
        bins = np.append(labels - 1, [np.inf])
        runtime_in_days = (data['maxt'] - data['mint']) / 86400.0
        data['days'] = pd.cut(runtime_in_days, bins=bins, labels=labels)
        data[TARGET_COL] = data[['job_type', 'days', 'fail']].apply(set_too_much_time, axis=1)
        data[CAT_COLS] = data[CAT_COLS].astype("category")
        
    def __remove_jobs_shorter_than_one_hour(self, data):
        print("--- Removing jobs with a duration less than an hour ---")
        data.drop(data[data['maxt'] - data['mint'] <= 3600].index, inplace=True)

    def __remove_missing_values(self, data):
        if data.isna().sum().sum() > 0:
            print("--- Removing records with missing values ---")
            data.dropna(inplace=True)
            
    def __remove_duplicated_jobs(self, data):
        print(f"--- Removing {(data['job'].duplicated() * 1).sum()} duplicated records ---")
        data.drop_duplicates(subset=['job'], inplace=True)
            
    def __zero_pad_and_truncate(self, x, max_length):
        length = len(x)
        if length < max_length:
            return np.pad(x, (0, max_length - length), mode='constant')
        else:
            return x[:max_length]
    
    def __pad_columns(self, df, columns, max_length):
        for col in columns:
            df[col] = df[col].apply(self.__zero_pad_and_truncate, args=(max_length,))
            
    def __transform_arrays_to_format(self, df, format_type):
        if format_type == "rows":
            return self.__transform_arrays_to_rows(df)
        else:
            return self.__transform_time_series_to_cols(df)
    
    def __transform_arrays_to_rows(self, df, max_length=96):
        print(f"--- Transforming arrays to rows ({len(df) * max_length}, features)")
        self.__pad_columns(df, [*TIME_SERIES_COLS, TIME_STEP_COL], max_length)
        df = df.explode([*TIME_SERIES_COLS, TIME_STEP_COL])
        df[TIME_SERIES_COLS] = df[TIME_SERIES_COLS].astype('float64')
        df[TIME_STEP_COL] = (np.arange(0, len(df)) % max_length)
        return df.sort_values(['job', TIME_STEP_COL]).reset_index(drop=True)
    
    def __transform_time_series_to_cols(self, df, max_length=96):
        self.__pad_columns(df, [*TIME_SERIES_COLS, TIME_STEP_COL], max_length)
        return pd.concat([
            df.drop([*TIME_SERIES_COLS, TIME_STEP_COL], axis=1).reset_index(drop=True), 
            pd.concat([pd.DataFrame(df[col].tolist()).astype('float64').add_prefix(f"{col}_") for col in TIME_SERIES_COLS], axis=1)
        ], axis=1)
    
    def __random_undersample(self, data, percentage):
        class_counts = data[TARGET_COL].value_counts()
        most_represented_class = class_counts.idxmax()
        most_represented_class_count = class_counts.max()
        
        samples_to_keep_count = int(most_represented_class_count * percentage)
        print(f"--- Undersampling the most represented class from {most_represented_class_count} to {samples_to_keep_count} ---")
        
        samples_to_remove = data[data[TARGET_COL] == most_represented_class]
        samples_to_keep = resample(samples_to_remove, replace=False, n_samples=samples_to_keep_count,
                                   random_state=self.random_state)

        data.drop(samples_to_remove.index, inplace=True)
        return pd.concat([data, samples_to_keep], ignore_index=True).reset_index(drop=True)
    
    def __transform_to_tensor(self, data):
        print(f"--- Transforming matrix to tensor ({len(data) // 96}, 96, features) ---")
        return np.stack(np.split(data, len(data) // 96))