import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#read in the dataframe with ISO encoding for some invalid char
#present in data like emojis adn other unwanted chars
df = pd.read_csv('../data/spam.csv',encoding='ISO-8859-1')

#drop unnecesary columns along axis 1
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

#rename columns
df.columns = ['labels','data']

#create binary labels
df['b_labels'] = df['labels'].map({'ham':0,'spam':1})
Y = df['b_labels'].values

#feature extraction from raw data
#can use tfidfVectorizer or CountVectorizer
countvectorizer = TfidfVectorizer(decode_error='ignore')
#countvectorizer = CountVectorizer(decode_error='ignore')
X = countvectorizer.fit_transform(df['data'])

#split up data
#can try cross validation?
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.33)

#model creation and training
model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("training score:",model.score(Xtrain,Ytrain))
print("test score:",model.score(Xtest,Ytest))

#visulaize data
def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=800,height=600).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

#visualize('spam')
#visualize('ham')

#create a new column predictions for testing model
df['predictions'] = model.predict(X)

#create a sneaky spam variable to find things that must be spam but aren't
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print("SNEAKY_SPAM")
    print(msg)

#create a sneaky spam variable to find things that must be spam but aren't
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print("NOT_ACTUALLY_SPAM")
    print(msg)
