from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report,f1_score,accuracy_score,confusion_matrix,roc_auc_score,roc_curve
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import string,re,os
import pandas as pd
from database import Database

STOPWORDS=stopwords.words('english')
WLM=WordNetLemmatizer()

def first_step_preprocess(text):
    text=text.lower().strip()
    text=re.compile('<.*?>').sub('',text)
    text=re.compile('[%s]' % re.escape(string.punctuation)).sub('',text)
    text=re.sub('\s+',' ',text)
    text=re.sub(r'\[[0-9]*\]',' ',text)
    text=re.sub(r'[^\w\s]','',str(text).lower().strip())
    text=re.sub(r'\d',' ',text)
    text=re.sub(r'\s+',' ',text)
    return text

def remove_stopwords(text):
    words=[word for word in text.split() if word not in STOPWORDS]
    return ' '.join(words)

def lemmatizer(text):
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    word_pos_tags=nltk.pos_tag(word_tokenize(text))
    a=[WLM.lemmatize(tag[0],get_wordnet_pos(tag[1])) for tag in word_pos_tags]
    return " ".join(a)

def preprocess(text):
    return lemmatizer(remove_stopwords(first_step_preprocess(text)))

if __name__=='__main__':
    database_path='./data/digital_rates.db'
    table_name='Company'
    
    db=Database(db_name='')
    company_descriptions=db.fetch_all_as_dataframe('Company')
    
    # Apply transformations
    traindf,testdf=train_test_split(company_descriptions,test_size=0.2,random_state=42,shuffle=True,stratify=company_descriptions['sector'])
    traindf['description']=traindf['description'].apply(lambda x:preprocess(x))
    
    
    # clf=Pipeline(
    #     [
    #         ('vect',CountVectorizer()),
    #         ('tfidf',TfidfTransformer()),
    #         ('clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42,max_iter=5,tol=None))
    #     ]
    # )
