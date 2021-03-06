import pandas as pd
import os
import json
import numpy as np

def check_size_prep_test(predict_raw_data_path):
    predict_df = pd.read_csv(predict_raw_data_path)
    if len(predict_df) < 100:
        raw_data_path = os.path.join(os.path.pardir,"data","raw", "heart_failure_clinical_records_dataset.csv")
        raw_df = pd.read_csv(raw_data_path)
        predict_df = pd.concat([raw_df,predict_df],axis=0)
    return predict_df
        
    
def identify_model_features(df):
    model_features = df.columns.drop('DEATH_EVENT')
    model_target = 'DEATH_EVENT'
    numerical_features_all = df[model_features].select_dtypes(include=np.number).columns
    categorical_features_all = df[model_features].select_dtypes(include='object').columns
    return numerical_features_all,categorical_features_all,model_target

def drop_outliers(df,numerical_features_all):
    for c in numerical_features_all:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3-Q1
        dropIndexes = df[df[c] < Q1-1.5*IQR].index
        if len(dropIndexes) > 0:
            df.drop(dropIndexes,inplace=True)
        dropIndexes = df[df[c] > Q3+1.5*IQR].index
        if len(dropIndexes) > 0:
            df.drop(dropIndexes,inplace=True)
    return df

def impute_numerical_features(df,numerical_features_all,numerical_imputed_json_path):
    df_imputed = df.copy()
    df_imputed[numerical_features_all] = df_imputed[numerical_features_all].fillna(df_imputed[numerical_features_all].mean())
    #store imputataion value in json
    numerical_imputation_json = df_imputed[numerical_features_all].mean().to_json(orient="index")
    with open(numerical_imputed_json_path,"w") as jsonfile:
        jsonfile.write(numerical_imputation_json)
    return df_imputed


def impute_categorical_features(df_imputed,categorical_features_all,categorical_imputed_json_path):
    mode_dict = dict()
    #identify the mode
    for c in df_imputed[categorical_features_all]:
        mode_value = df_imputed[c].mode()
        mode_dict[c] = mode_value

        #impute feature with mode
        df_imputed[c].fillna(mode_value,inplace=True)
    #store ctegorical mode values in json
    categorical_imputation_json = json.dumps(mode_dict)
    with open(categorical_imputed_json_path,"w") as jsonfile:
        jsonfile.write(categorical_imputation_json)
    return df_imputed

def feature_encoding(df,model_target):
    df = df.drop(columns=["time"])
    
    df = df.astype({"age":"int8"})
    #cut equal size bins for age with actual frequency..use qcut where unequal bins but distribution of value

    age_bins=[0,5,15,30,45,60,75,90,105]
    age_labels = ["kid","teen","young","midage","upperage","senior","old","veryold"]
    df["age"] = pd.cut(df["age"],bins=age_bins,labels=age_labels)
    qcut_labels = ["verylow","low","med","high","veryhigh"] 
    df["creatinine_phosphokinase"] = pd.qcut(df["creatinine_phosphokinase"],q=5,labels=qcut_labels)
    #df["ejection_fraction"] = pd.qcut(df["ejection_fraction"]),q=5,lables=qcut_labels)
    df["platelets"] = pd.qcut(df["platelets"],q=5,labels=qcut_labels)
    df["serum_creatinine"]=pd.qcut(df["serum_creatinine"],q=5,labels=qcut_labels)
    #df["serum_sodium"] = pd.qcut(df["serum_sodium"],q=5,labels=qcut_labels)
    #df["time"] = pd.qcut(df["time"],q=5,labels=qcut_labels)
    df_encoded = pd.get_dummies(df)
    #move target column to the end of df
    target = df_encoded[model_target]
    df_encoded.drop(columns=[model_target], inplace=True)
    df_encoded.insert(30,model_target,target)
    return df_encoded