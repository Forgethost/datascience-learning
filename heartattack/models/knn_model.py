import pandas as pd
import os
import sys
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score,accuracy_score

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
        df.drop(dropIndexes,inplace=True)
        dropIndexes = df[df[c] > Q3+1.5*IQR].index
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

def feature_encoding(df_imputed,model_target):
    df = df_imputed.drop(columns=["time"])
    
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

if __name__ == "__main__":
    raw_data_path = os.path.join(os.path.pardir,"data","raw", "heart_failure_clinical_records_dataset.csv")
    prediction_path = os.path.join(os.path.pardir,"data","predictions", "prediction.csv")
    trained_model_path = os.path.join(os.path.pardir,"models","knn_model.pickle")
    numerical_imputed_json_path = os.path.join(os.path.pardir,"static_data","numeric_imputed_values.json")
    categorical_imputed_json_path = os.path.join(os.path.pardir,"static_data","categorical_imputed_values.json")
    
    try:
        dataset = sys.argv[-1]
        test_dataset_path = os.path.join(os.path.pardir,"data","raw", dataset)    
    except:
        pass
    print("Exception orrcured...using sample datatset")
        #raw_data_path = os.path.join(os.path.pardir,"data","raw", "heart_failure_clinical_records_dataset.csv")
        
    df = pd.read_csv(raw_data_path)
    #separating feature and target  
    numerical_features_all, categorical_features_all, model_target = identify_model_features(df)
    
    #removing outliers
    df = drop_outliers(df,numerical_features_all)
    #Impute numerical feature missing with mean value
    df_imputed = impute_numerical_features(df,numerical_features_all,numerical_imputed_json_path)

    #Impute categorical feature missing with Mode (most frequent) value
    df_imputed = impute_categorical_features(df_imputed,categorical_features_all,categorical_imputed_json_path)
    #Feature encoding
    df_encoded = feature_encoding(df_imputed,model_target)

    train_data, test_data = train_test_split(df_encoded,test_size=0.1,shuffle=True,random_state=23)
    class_0 = train_data[train_data[model_target] == 0 ]
    class_1 = train_data[train_data[model_target] == 1 ]

    sampled_class_1 = class_1.sample(n=len(class_0),replace=True,random_state=42)

    train_data = pd.concat([sampled_class_1,class_0])
    train_data = shuffle(train_data)
    #use pipeline classsifier
    final_model_features = train_data.iloc[0:1,0:31].columns
    X_train = train_data[final_model_features]
    y_train = train_data[model_target]

    X_test = test_data[final_model_features]
    y_test = test_data[model_target]
    classifier = Pipeline([('imputer',SimpleImputer(strategy='mean')),
                            ('estimator',KNeighborsClassifier(n_neighbors=6,metric="manhattan"))   
                        ])
    
    knn_clf = classifier.fit(X_train,y_train)
    
    #predict
    train_prediction = classifier.predict(X_train)
    #Model perforfance and report for Train data
    print("Model perforfance for Train data", confusion_matrix(y_train,train_prediction),end="\n")
    print(classification_report(y_train,train_prediction),end="\n")
    print("Model accuracy train data: ",accuracy_score(y_train,train_prediction),end="\n")
    
    #Model performance for Test data
    X_test = test_data[final_model_features]
    y_test=test_data[model_target]
    test_prediction = classifier.predict(X_test)

    print("Model perforfance ", confusion_matrix(y_test,test_prediction),end="\n")
    print(classification_report(y_test,test_prediction),end="\n")
    print("Model accuracy: ",accuracy_score(y_test,test_prediction),end="\n")
    #pd.concat([test_data.reset_index(),pd.DataFrame(test_prediction).reset_index()],axis=1).to_csv(prediction_path,header=True,index=False)
    #save trained model
    with open(trained_model_path, "wb") as trained_pkl_file:
        pickle.dump(knn_clf,trained_pkl_file)