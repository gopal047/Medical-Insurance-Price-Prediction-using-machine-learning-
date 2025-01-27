import pandas as pd 

import numpy as np 

import seaborn assns

import matplotlib. pyplot as plt

from matplotlib. figure import Figure

df= pd.read_csv('C:/Users/Syber Computers/Downloads/insurance.csv') 

df.head()

from sklearn.preprocessing import LabelEncoder 

le=LabelEncoder()

le.fit(df.sex.drop_duplicates()) 

df.sex=le.transform(df.sex)

#smoker 

le.fit(df.smoker.drop_duplicates()) 

df.smoker=le.transform(df.smoker) 

#region 

le.fit(df.region.drop_duplicates()) 

df.region=le.transform(df.region) 

corr=df.corr().round(2) 

plt.figure(figsize=(15,15))

sns. heatmap(corr,annot=True,cmap='crest')

sns.catplot(x="sex",y="expenses",hue="smoker", 

kind="violin",data=df,palette='magma')

<seaborn.axisgrid.FacetGrid at 0x14b6c198b50>
plot_columns=['age', 'bmi'] 

def box_plot(data, columns):

fig, axes = plt.subplots(nrows=len(columns), ncols=1, 

figsize=(20,25))

for i, col in enumerate(columns): 

sns.boxplot(data[col], ax=axes[i]) 

axes[i].set_xlabel(col)

plt.tight_layout() 

plt.show()

box_plot(df,plot_columns)

sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", 

data=df)

<seaborn.axisgrid.FacetGrid at 0x14b62b0cb90> 

sns.catplot(x="smoker", kind="count",hue = 'sex', palette="rainbow", 

data=df[(df.age == 18)])

plt.title("The number of smokers and non-smokers (18 years old)") 

Text(0.5, 1.0, 'The number of smokers and non-smokers (18 years old)') 

sns.barplot(data=df, x="smoker", y="expenses",hue="smoker")

<Axes: xlabel='smoker', ylabel='expenses'

sns.lmplot(x="age", y="expenses", hue="smoker", data=df, palette = 

'inferno_r')

ax.set_title('Smokers and non-smokers') 

Text(0.5, 1.0, 'Smokers and non-smokers')

plt.figure(figsize=(12,5))

plt.title("Distribution of expenses for patients with BMI greater than 

30")

ax = sns.distplot(df[(df.bmi >= 30)]['expenses'])

#preparing the data
pip3 install xgboost 

import xgboost

Requirement already satisfied: xgboost in c:\users\syber computers\

anaconda\lib\site-packages (2.1.0)

Requirement already satisfied: numpy in c:\users\syber computers\

anaconda\lib\site-packages (from xgboost) (1.24.3)

Requirement already satisfied: scipy in c:\users\syber computers\

anaconda\lib\site-packages (from xgboost) (1.11.1)

from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LinearRegression 

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor 

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split 

from sklearn.metrics import

mean_absolute_error,mean_squared_error,r2_scoreX=df.drop(['expenses'],axis=1) 

y=df.expenses

age sex bmi children smoker region 

results=pd.DataFrame(columns=['MAE','MSE','R2-score']) 

for method,func in regressors.items():

model = func.fit(X_train,y_train) 

pred = model.predict(X_test) 

results.loc[method]=

[np.round(mean_absolute_error(y_test,pred),3), 

np.round(mean_squared_error(y_test,pred),3), 

np.round(r2_score(y_test,pred),3)

]

results.sort_values('R2-

score',ascending=False).style.background_gradient(cmap='Greens',subset
=['R2-score'])

<pandas.io.formats.style.Styler at 0x14b6fb1a250>
