import pandas as pd

from sklearn import svm
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
import pickle 
#loading data
df = pd.read_csv("roman_urdu.csv")
df.columns = ['comment','sentiment', 'nan']
# dropping the columns and missing values 
df.drop(columns = ['nan'], axis = 1, inplace=True)
df = df.dropna()
i = df[df['sentiment'] == 'Neative'].index
df.drop(i, inplace = True)

#text preprocessing
def text_process(text):
  return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",str(text).lower()).split())

df['processed_text'] = df["comment"].apply(text_process)

# remove stopwords
stopwords=['ai', 'ayi', 'hy', 'hai', 'main', 'ki', 'tha', 'koi', 'ko', 'sy', 'woh', 'bhi', 'aur', 'wo', 'yeh', 'rha', 'hota', 'ho', 'ga', 'ka', 'le', 'lye', 'kr', 'kar', 'lye', 'liye', 'hotay', 'waisay', 'gya', 'gaya', 'kch', 'ab', 'thy', 'thay', 'houn', 'hain', 'han', 'to', 'is', 'hi', 'jo', 'kya', 'thi', 'se', 'pe', 'phr', 'wala', 'waisay', 'us', 'na', 'ny', 'hun', 'rha', 'raha', 'ja', 'rahay', 'abi', 'uski', 'ne', 'haan', 'acha', 'nai', 'sent', 'photo', 'you', 'kafi', 'gai', 'rhy', 'kuch', 'jata', 'aye', 'ya', 'dono', 'hoa', 'aese', 'de', 'wohi', 'jati', 'jb', 'krta', 'lg', 'rahi', 'hui', 'karna', 'krna', 'gi', 'hova', 'yehi', 'jana', 'jye', 'chal', 'mil', 'tu', 'hum', 'par', 'hay', 'kis', 'sb', 'gy', 'dain', 'krny', 'tou']
def remove_stopwords(text):
  word_tokens = word_tokenize(text)
  filtered_sentence = [w for w in word_tokens if not w in stopwords]
  return " ".join(filtered_sentence)

df['removed_stopwords'] = df['processed_text'].apply(remove_stopwords)

# label encoding
y = df.iloc[:,1].values
LE_y = LabelEncoder()
y = LE_y.fit_transform(y)

vectorizer = TfidfVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df.removed_stopwords,y, test_size = 0.20, random_state = 0)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
#print (vectorizer.vocabulary_)

# using Support Vector Classifier as the classification algorithm
svc_clf = svm.SVC(gamma = 'scale', random_state = 0, C = 10429215.975263635)
svc_clf.fit(x_train, y_train)
#pickle the model

vec_pickle = open('vector_pickle', "wb")
pickle.dump(vectorizer, vec_pickle)
vec_pickle.close()

model_pickle = open('model_pickle', "wb")
pickle.dump(svc_clf, model_pickle)
model_pickle.close()







