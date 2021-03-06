# Create Bi/Tri-grams
by: Pratyush Singh

Create bi/tri-grams out of column data. The bi/trigram are **created for each row within the data**. This application is different than other use-cases where bi/trigrams are created for the entire text as a whole. In this case, a bi/tri-gram is created for each row within the data after pre-processesing. The list of bi/tri-grams returned are useful for visualization of your data through WordClouds or other word frequency visualizations.

## Requirements
1. python 3.*
2. pip3

## Installation
`pip3 install -r requirements.txt`

### Installation in Jupyter Notebooks
```
!pip install -r https://raw.githubusercontent.com/pratyushsingh97/Bigrams/master/requirements.txt
!curl -O https://raw.githubusercontent.com/pratyushsingh97/Bigrams/master/bigrams.py
```

## How to Run
Simply pass the column of text within your DataFrame to the `runner()` function. The `runner()` function returns a list of n-grams. 
### Adding Stopwords
`bigrams.py` already filters out for common stopwords; however, based on your data you may want to add additional filtering. You can pass a list of words that you would like to filter, and `bigrams.py` adds these words to the list of stopwords.
### Example
```python
import pandas as pd

dummy_data = pd.DataFrame({"input_text": ["I like pie", 
                                          "I like to eat", 
                                          "I like to watch movies on the weekends"]})
bigrams = runner(dummy_data['input_text'], stopwords=["like"]) # add "like" to the list of stopwords

print(bigrams) # prints a list of three bigrams (one for each row)
```

### Usage in Jupyter Notebook
*Make sure to follow the installation steps for Jupyter Notebooks.*
```python
import bigrams
import pandas as pd

dummy_data = pd.DataFrame({"input_text": ["I like pie", 
                                          "I like to eat", 
                                          "I like to watch movies on the weekends"]})
bigrams = bigrams.runner(dummy_data['input_text'], stopwords=["like"]) # add "like" to the list of stopwords
print(bigrams) # prints a list of three bigrams (one for each row)
```

## Additional Details
1. `bigrams.py` utilizes the nltk library to score each bi/tri-gram created for each input text. The highest rated bi/tri-gram is returned. If no bi/tr-grams exist within the data, then the original text is returned.
2. `bigrams.py` lemmatizes the words in the input text, so similar phrases will lead to the same bigram. For example "I am eating pie" and "I eat pie" result in the same bigram "eat_pie".
3. This function only works on `pandas.core.series.Series` objects right now. In most cases, this is equivalent to the column within your pandas DataFrame you wish to analyze (i.e. `dummy_data['input_text']`).


