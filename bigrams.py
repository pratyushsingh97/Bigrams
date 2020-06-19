
import pandas as pd
import nltk
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stopset = set(stopwords.words('english'))
stops = [word for word in stopwords.words('english')]
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
wordnet_lemmatizer = WordNetLemmatizer()

def _stopwords(extend_stopwords=None) -> None:
    if type(extend_stopwords) == list:
        stops.extend(extend_stopwords)
    if type(extend_stopwords) == str:
        stops.append(extend_stopwords)
    
def _collocations(text:str) -> str:
    """This is a helper function that creates bigrams out of the text.
    
    Collocations are bigrams that are paired together based on a
    similarity score. This is an effort to construct useful bigrams.
    
        Args:
            text: string that needs to be turned into bigrams
        
        Returns:
            str: returns the bigrams if found or returns the unigram text
    """
    filter_stops = lambda w: len(w) < 3 or w in stops
    uncovered_words = [word for word in word_tokenize(text) if word.lower() not in stops]
    uncovered_words = [wordnet_lemmatizer.lemmatize(word) for word in uncovered_words]
    finder = BigramCollocationFinder.from_words(uncovered_words)
    finder.apply_word_filter(filter_stops)
    bigram = finder.nbest(bigram_measures.pmi, 1)
    
    if not bigram:
        return text
    
    return f"{bigram[0][0]}_{bigram[0][1]}"

def _create_bigrams(col: pd.Series) -> list:
    """ Helper function to take the column of text and return the bigrams 
    based on the input text
    
    Args:
        df: column of text such as df['input_text']
    
    Returns:
        bigrams: list of bigrams and in some cases unigrams (when bigrams don't exist)
    """
    
    if type(col) != pd.Series:
        raise TypeError("The parameter passed must be a Series object.Pass the column you wish to create n-grams out of.")
    try:
        bigrams = []
        for index, value in col.items():
            bigram = _collocations(value)
            bigrams.append(bigram)

        return bigrams
    except Exception as e:
        print(e)
    
def runner(col: pd.Series, stopwords=None):
    _stopwords(stopwords)
    return _create_bigrams(col)
