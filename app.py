from flask import Flask,render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from nltk.tokenize import word_tokenize


app = Flask(__name__)

model = pickle.load(open('model_pickle', 'rb'))
vec = pickle.load(open('vector_pickle','rb'))

@app.route('/')
def form():
    return render_template('sentiment.html')

def text_process(text):
  return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",str(text).lower()).split())
stopwords=['ai', 'ayi', 'hy', 'hai', 'main', 'ki', 'tha', 'koi', 'ko', 'sy', 'woh', 'bhi', 'aur', 'wo', 'yeh', 'rha', 'hota', 'ho', 'ga', 'ka', 'le', 'lye', 'kr', 'kar', 'lye', 'liye', 'hotay', 'waisay', 'gya', 'gaya', 'kch', 'ab', 'thy', 'thay', 'houn', 'hain', 'han', 'to', 'is', 'hi', 'jo', 'kya', 'thi', 'se', 'pe', 'phr', 'wala', 'waisay', 'us', 'na', 'ny', 'hun', 'rha', 'raha', 'ja', 'rahay', 'abi', 'uski', 'ne', 'haan', 'acha', 'nai', 'sent', 'photo', 'you', 'kafi', 'gai', 'rhy', 'kuch', 'jata', 'aye', 'ya', 'dono', 'hoa', 'aese', 'de', 'wohi', 'jati', 'jb', 'krta', 'lg', 'rahi', 'hui', 'karna', 'krna', 'gi', 'hova', 'yehi', 'jana', 'jye', 'chal', 'mil', 'tu', 'hum', 'par', 'hay', 'kis', 'sb', 'gy', 'dain', 'krny', 'tou']

def remove_stopwords(text):
  word_tokens = word_tokenize(text)
  filtered_sentence = [w for w in word_tokens if not w in stopwords]
  return " ".join(filtered_sentence)

@app.route('/predict', methods=['POST','GET'])
def predict():
    comment = request.form['comment'] # no more problem
    user_refined = text_process(comment)# no more problem
    clean_text = remove_stopwords(user_refined)# no more problem
    clean_text = [clean_text]#no more problem
    
    value_tfidf = vec.transform(clean_text)
    prediction = model.predict(value_tfidf)
    
    if prediction == 0:
        return render_template("sentiment.html", pred='NEGATIVE' )
    elif prediction == 1:
        return render_template("sentiment.html", pred='NEUTRAL')
    else:
        return render_template("sentiment.html", pred='POSITIVE')
    
    
    
    

    
if __name__  == '__main__':
    app.run(debug=True)
    
    
    
