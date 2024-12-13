import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
nltk.download('punkt')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv('labeled.csv', sep=',')

train_df, test_df = train_test_split(df, test_size=500)

def tokenize_sentence(sentence: str, remove_stop_words: bool=True):

    snowball = SnowballStemmer(language='russian')
    russian_stop_words = stopwords.words('russian')

    tokenizer = RegexpTokenizer(r'\b\w+\b')
    tokens = tokenizer.tokenize(sentence.lower())
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]

    return tokens

model_pipeline_c_10 = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=tokenize_sentence)),
    ('model', LogisticRegression(random_state=0, C=10.))
    ])

model_pipeline_c_10.fit(test_df['comment'], test_df['toxic'])

joblib.dump(model_pipeline_c_10, 'comment_classificat_model.pkl')