import re
import nltk
import pandas as pd
from nltk.stem.isri import ISRIStemmer
from farasa.stemmer import FarasaStemmer

 
### Clean each tweet
def clean_data(data,stemming=False):
    newList=[]
    for i, produto in data.iterrows():
        # remove the Links
        newTweet=re.sub(r'http\S+', '', produto['text'])
        # remove english letters
        newTweet=re.sub(r'[a-zA-Z]', '', newTweet)
        # remove numbers
        newTweet=re.sub(r'[0-9]', '', newTweet)
        # remove the arabic numbers
        newTweet=re.sub(r'[\u0660-\u0669]', '', newTweet)
        # remove the emails
        newTweet=re.sub(r'\S*@\S*\s?', '', newTweet)
        # remove the hashtags
        newTweet=re.sub(r'#\S+', '', newTweet)
        # remove the mentions
        newTweet=re.sub(r'@\S+', '', newTweet)
        # remove emojis
        RE_EMOJI = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                        "]+", re.UNICODE)
        newTweet=RE_EMOJI.sub(r'', newTweet)
        # replace _ by whitespace
        newTweet=re.sub(r'_', ' ', newTweet)
        
        # remove the punctuations
        newTweet = re.sub('\W+',' ', newTweet)

        # remove duplicated whitespaces
        newTweet=re.sub(r'\s+', ' ', newTweet)

        # applay tokenization
        newTweet=newTweet.split()

        # may not be necessary بتبوظ الكلام
        
        # remove stopwords
        stopwords_arabic = nltk.corpus.stopwords.words('arabic')
        newTweet=[word for word in newTweet if word not in stopwords_arabic]
        # join the words to form a sentence
        newTweet=' '.join(newTweet)

        if stemming:
            # apply stemming
            # farasa stemmer take the whole array of words
            stemmer = FarasaStemmer()
            newTweet=stemmer.stem(newTweet)

        newList.append(newTweet)
    data['text']=newList
    return data

