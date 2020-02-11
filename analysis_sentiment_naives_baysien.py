import pandas as pd
import nltk
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

df=pd.read_csv(sys.argv[1],sep='\t',names=['label' ,'sentiment'])
vectorizer=TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii')
y=df.label
x=vectorizer.fit_transform(df.sentiment)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
print("ok")
clf=naive_bayes.MultinomialNB()
clf.fit(x_train,y_train)
accuracy=roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])
print("the accuracy is {}".format(accuracy))