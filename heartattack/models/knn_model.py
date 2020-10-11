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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score,accuracy_score
from model_functions import *

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
    final_model_features = train_data.iloc[0:1,0:30].columns
    X_train = train_data[final_model_features]
    y_train = train_data[model_target]

    X_test = test_data[final_model_features]
    y_test = test_data[model_target]
    
    classifier = Pipeline([('imputer',SimpleImputer(strategy='mean')),
                           #('dt',DecisionTreeClassifier(criterion='gini')),
                           ('clf',RandomForestClassifier(n_estimators=100,max_samples=None, max_features='auto',criterion='gini'))
                            #('estimator',KNeighborsClassifier(n_neighbors=8,metric="manhattan"))   
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