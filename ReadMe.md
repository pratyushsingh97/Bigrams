# Create Bigrams
by: Pratyush Singh

Create bi/tri-grams out of column data. The bi/trigram are **created for each row within the data**. This application is different than other use-cases where bi/trigrams are created for the entire text as a whole. In this case, a bi/tri-gram is created for each rowwithin the data after pre-processesing.

## Requirements
1. python 3.*
2. pip

## Installation
`pip install -r requirements.txt`

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

print(bigrams)
```

## Additional Details
1. `bigrams.py` utilizes the nltk library to score each bi/tri-gram created for each input text. The highest rated bi/tri-gram is returned. If no bi/tr-grams exist within the data, then the original text is returned.


