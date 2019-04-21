import matplotlib.pyplot as plt
from time import time
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    tok = WordPunctTokenizer()
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    combined_pat = r'|'.join((pat1, pat2))
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig") \
                        .replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, 
    # it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

def main():
    cols = ['sentiment','id','date','query_string','user','text']
    df = pd.read_csv('Data/training.1600000.processed.noemoticon.csv',
                     header=None, names=cols)
    # above line will be different depending on 
    # where you saved your data, and your file name
    print(df.head())
    print(df.sentiment.value_counts())
    df.drop(['id','date','query_string','user'],axis=1,inplace=True)
    print(df[df.sentiment == 0].head(10))
    print(df[df.sentiment == 4].head(10))

    df['pre_clean_len'] = [len(t) for t in df.text]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.boxplot(df.pre_clean_len)
    plt.savefig("str_len_distribution.pdf")

    df[df.pre_clean_len > 140].head(10)

    print("Cleaning and parsing the tweets...\n")
    clean_tweet_texts = []
    for i in xrange(df.shape[0]):
        if (i+1) % 10000 == 0:
            print "Tweets %d of %d has been processed" %(i+1, df.shape[0])                                                                 
        clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

    clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
    clean_df['target'] = df.sentiment
    print(clean_df.head())

    clean_df.to_csv('Data/processed_tweets.csv',encoding='utf-8')
    csv = 'Data/processed_tweets.csv'
    my_df = pd.read_csv(csv,index_col=0)
    print(my_df.head())


if __name__ == "__main__":
    main()
