import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.pipeline import Pipeline

df = pd.read_csv("Heart_Disease_Prediction.csv")
x = df.drop(["Heart Disease"], axis = 1)
y = df["Heart Disease"].values
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.32)

scaler = StandardScaler() 
print('\nData preprocessing with {scaler}\n'.format(scaler=scaler)) 
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

mlp = MLPClassifier(
       max_iter=1000,
       alpha=0.1,
       random_state=42
      )
mlp.fit(X_train_scaler, y_train)

mlp_predict = mlp.predict(X_test_scaler)
accuracy=accuracy_score(y_test, mlp_predict) * 100

print('MLP Accuracy: {:.2f}%'.format(accuracy))