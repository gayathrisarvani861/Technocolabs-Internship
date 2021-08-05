import pickle

import nltk
import numpy as np
import pandas as pd
import streamlit as st

nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy import stats

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Adding Subjectivity and Polarity columns
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob



def getsubj(text):
  return TextBlob(text).sentiment.subjectivity
def getpolarity(text):
  return TextBlob(text).sentiment.polarity

 
from PIL import Image

pickle_in = open("linearmodel1.pkl","rb")
linear = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

# Functions to predict Close
import string


def char_rmvl(text):                 #Removing all char except a-z and A-Z and replace them with ' '
    new=[char for char in text if char not in string.punctuation]
    new_str=''.join(new)
    new.clear()
    return new_str

stop = stopwords.words('english')

#Apply lemmatization
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer  # used to perform lemmatization
from nltk.tokenize import word_tokenize


def lemmat(text):
    lemma=WordNetLemmatizer()
    words=word_tokenize(text)
    return ' '.join([lemma.lemmatize(word) for word in words])

sid = SentimentIntensityAnalyzer()

def filter(Headlines):
    Head = [ x.lower() for x in Headlines]
    Head = [char_rmvl(x) for x in Head]
    Head = [' '.join([word for word in s.split() if word not in (stop)]) for s in Head]
    Head = [lemmat(s) for s in Head]
    return Head

def sent_anls(Head):
    compound = [sid.polarity_scores(x)['compound'] for x in Head]
    negative = [sid.polarity_scores(x)['neg'] for x in Head]
    neutral = [sid.polarity_scores(x)['neu'] for x in Head]
    positive = [sid.polarity_scores(x)['pos'] for x in Head]
    subjectivity = [getsubj(x) for x in Head]
    polarity = [getpolarity(x) for x in Head]

    return compound,negative,neutral,positive,subjectivity,polarity




def main():
    st.title("--------STOCK PRICE PREDICTION--------")


    
    html_temp = """
    <div style="background-color:#BB5BE7;padding:10px">
    <h2 style="color:white;text-align:center;font-family:'Tahoma';font-weight:bolder;">QUALCOMM Close Price Predictor</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)



    Open = st.text_input("OPEN")
    High = st.text_input("HIGH")
    Low = st.text_input("LOW")
    Volume = st.text_input("VOLUME")
    Headlines = st.text_input("HEADLINES")
    Headlines = list(Headlines.split("-"))
    head = filter(Headlines)
    cmpd,negt,neut,post,subj,pol = sent_anls(head)
    
    
    #compound = st.slider("Compound",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    #negative = st.slider("negative",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    #eutral = st.slider("neutral",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    #positive = st.slider("positive",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    #Subjectivity = st.slider("Subjectivity",min_value=0.00 , max_value = 1.00 ,step = 0.01)
    #Polarity = st.slider("Polarity",min_value=0.00 , max_value = 1.00 ,step = 0.01)

    af = pd.DataFrame()
    af['compound'] = cmpd
    af['negative'] = negt
    af['neutral'] = neut
    af['positive'] = post
    af['Open'] = Open 
    af['High'] = High
    af['Low'] = Low
    af['Volume'] = Volume
    af['Subjectivity'] = subj
    af['Polarity'] = pol

    
    result=""

    if st.button("Predict"):
        result = linear.predict(af)[0]
    st.success('Predicted Close Price : $ {}'.format(result))
    

if __name__ =='__main__':
    main()