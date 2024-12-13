# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
     from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/bmi.csv

import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("drive/MyDrive/bmi.csv")

df.head()
![Screenshot 2024-12-13 125038](https://github.com/user-attachments/assets/cfc5142d-df2a-44de-b35d-b75a6049f717)

df.dropna()

max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
199
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
![Screenshot 2024-12-13 125024](https://github.com/user-attachments/assets/6dcd47d1-6725-4fbc-901c-7a75a8ac1a2f)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
![Screenshot 2024-12-13 125016](https://github.com/user-attachments/assets/eada6210-ee89-45e2-b778-bdd09b7446d2)

from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
![Screenshot 2024-12-13 125009](https://github.com/user-attachments/assets/0858ac4e-04f8-4675-b921-adb5b3497b89)

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
![Screenshot 2024-12-13 125001](https://github.com/user-attachments/assets/88ff6e78-04e5-466a-afea-ab121cdc3b8f)

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
![Screenshot 2024-12-13 124951](https://github.com/user-attachments/assets/93a10391-b7de-4c20-87d6-9f7c673e61de)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

df.columns
![Screenshot 2024-12-13 124940](https://github.com/user-attachments/assets/6e24e654-1796-46c0-bf92-e9b82f50f9f7)

df.shape
![Screenshot 2024-12-13 124931](https://github.com/user-attachments/assets/809dcab9-3340-4738-b7b4-5856dafb2b77)

x=df.drop('Survived',axis=1)
y=df['Survived']

df=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df.columns
![Screenshot 2024-12-13 124924](https://github.com/user-attachments/assets/d6e3d939-9fa0-4bbf-8438-a13a5055de94)

df['Age'].isnull().sum()
177

df['Age'].fillna(method='ffill')
![Screenshot 2024-12-13 124917](https://github.com/user-attachments/assets/8ed56ccb-229c-4b8c-b356-474437e7cf30)

df['Age']=df['Age'].fillna(method='ffill')
df['Age'].isnull().sum()
![Screenshot 2024-12-13 124902](https://github.com/user-attachments/assets/86342aed-38ee-48db-b9ea-4339dd3496df)

data=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x
![Screenshot 2024-12-13 124853](https://github.com/user-attachments/assets/fc7d079b-440d-47d2-9c37-7aac7c838a5b)

data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes

data
![Screenshot 2024-12-13 124841](https://github.com/user-attachments/assets/526d6b00-42ba-4b24-8451-6c7e8aff5fb8)

for column in['Sex','Cabin','Embarked']:
   if x[column].dtype=='object':
             x[column]=x[column].astype('category').cat.codes
k=5
selector=SelectKBest(score_func=chi2,k=k)
X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
![Screenshot 2024-12-13 124829](https://github.com/user-attachments/assets/534b0875-436a-4b83-a090-a2f30a0480f1)

x.info()
![Screenshot 2024-12-13 124821](https://github.com/user-attachments/assets/8e9956c4-0bc1-4242-ba15-45e2b5cba8ae)

x=x.drop(["Sex","Cabin","Embarked"],axis=1)
x
![Screenshot 2024-12-13 124814](https://github.com/user-attachments/assets/2730982c-1f2e-4927-b35a-b003370c84c9)

from sklearn.feature_selection import SelectKBest, f_regression
selector=SelectKBest(score_func=f_regression,k=5)
X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector=SelectKBest(score_func=mutual_info_classif,k=5)
X_new=selector.fit_transform(x,y)

selected_features_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
![Screenshot 2024-12-13 124758](https://github.com/user-attachments/assets/34c43cf9-d421-4cdc-b442-735348c45edb)

from sklearn.feature_selection import SelectPercentile,chi2
selector=SelectPercentile(score_func=chi2,percentile=10)
x_new=selector.fit_transform(x,y)
![Screenshot 2024-12-13 124805](https://github.com/user-attachments/assets/1e0adffd-eb1f-4715-8ef0-df767150fd30)

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
![Screenshot 2024-12-13 124751](https://github.com/user-attachments/assets/3868a2de-0a3b-4458-be4a-239106d76f2d)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance = model.feature_importances_
threshold=0.15
selected_features=x.columns[feature_importance > threshold]
print("Selected Features:")
print(selected_features)
![Screenshot 2024-12-13 124744](https://github.com/user-attachments/assets/065d1098-f0ab-4796-889a-af254fda54e4)

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")
df.columns
![Screenshot 2024-12-13 124739](https://github.com/user-attachments/assets/0533e115-9dd9-4f23-acf3-e40affdda7ba)

df
![Screenshot 2024-12-13 124732](https://github.com/user-attachments/assets/eca368a5-81e3-476f-8d7d-2a9243f9c992)

df.isnull().sum()
![Screenshot 2024-12-13 124715](https://github.com/user-attachments/assets/44f1e551-34db-4eb6-bdbb-e728f0191aa1)

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

df = pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()
![Screenshot 2024-12-13 124707](https://github.com/user-attachments/assets/0cacb507-f729-40e0-830a-4c1b68258641)

contigency_table=pd.crosstab(tips["sex"],tips["time"])
contigency_table
![Screenshot 2024-12-13 124701](https://github.com/user-attachments/assets/73e0fe88-065c-4465-95f0-9ca89c4e11dd)

chi2,p,_,_=chi2_contingency(contigency_table)
print(f"chi-Squared Statistic: {chi2}")
print(f"p-value: {p}")
![Screenshot 2024-12-13 124654](https://github.com/user-attachments/assets/9987787b-892a-4124-95b5-f12889b5fe6e)

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

data={
      'Feature1':[1,2,3,4,5],
      'Feature2':['A','B','C','A','B'],
      'Feature3':[0,1,1,0,1],
      'Target':[0,1,1,0,1]
      }
      df=pd.DataFrame(data)
      x=df[['Feature1','Feature3']]
      y=df['Target']
      selector = SelectKBest(score_func=f_classif, k=2)
      selector.fit(x, y)
      selector_feature_indices=selector.get_support(indices=True)
      selected_features=x.columns[selector_feature_indices]
      print("Selected Features:")
      print(selected_features)
      print("selected_Features:")
      print(selected_features) # Assuming selected_features holds the desired value
![Screenshot 2024-12-13 124646](https://github.com/user-attachments/assets/3a43ff79-4e90-4f19-a71d-7945128c871b)

# RESULT:
      The code executed successfully
