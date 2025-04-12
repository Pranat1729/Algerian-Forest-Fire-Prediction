import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as mp  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv(r'C:\Users\Pranat\Documents\CETPA\Algerian_forest_fires_dataset.csv')
#print(df.isnull().sum())
df.dropna()
df['Temperature'] = df['Temperature'].astype('int')
df['Classes'] = df['Classes'].map({'not fire':0, 'fire':1})
print(df.dtypes)
#print(df.head)
dfn = df.loc[:,['Temperature', 'RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI','Classes']]
dataplot = sb.heatmap(dfn.corr(), annot=True) 
mp.show()


#### according to the output of correlation heatmap, we pick out the columns which are most correlated with Classes and least correlated among themselves to perform regression analysis.
x = dfn[['FFMC','FWI']]
y = dfn['Classes']
def classify(model,x,y):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0, shuffle=True, stratify = y)

	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.transform(x_test)
	#print(x_train)
	model.fit(x_train,y_train)
	print('Accuracy:', model.score(x_test, y_test)*100)
	y_pred = model.predict(x_test)
	cm = confusion_matrix(y_test, y_pred)
	print(cm)

model = LogisticRegression(multi_class='multinomial') ## solver should be lbfgs
classify(model, x, y)
#print(dfn.head)