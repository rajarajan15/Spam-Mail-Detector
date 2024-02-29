import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#This is the data collection part
raw_data=pd.read_csv('/content/mail_data.csv')
print(raw_data)
#to replace null values with null string
mail_data=raw_data.where((pd.notnull(raw_data)),'')
#Print the head of dataframe
mail_data.head()
#to check total number of rows
mail_data.shape
#Spam mail is 0 and normal one as 1 
mail_data.loc[mail_data['Category'] =='spam','Category']=0
mail_data.loc[mail_data['Category'] =='ham','Category']=1
#Seperating the texts and label 
X = mail_data['Message']
Y = mail_data['Category']
print(X)
print(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
#transforming the text data to feature vectors which is input for logistic regression model
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
print(X_train_features)#it prints numerical values because of tfidvectorization and used for model
model = LogisticRegression()#Loaded the model
#training model with training data
model.fit(X_train_features,Y_train)
#predicting the training data
prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
print('Accuracy:',accuracy_on_training_data)
#predicting the twst data
prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)
print('Accuracy:',accuracy_on_test_data)
input_mail=["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]
#text to feature vectors
input_data_features = feature_extraction.transform(input_mail)
#make predictions
pred=model.predict(input_data_features)
print(pred)

if(pred[0]==1):
  print('Valid Mail')
else:
  print('Spam Mail')