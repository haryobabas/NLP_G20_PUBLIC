from wordcloud import WordCloud
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


stop_words_indo = set(stopwords.words('indonesian'))
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
pat3 = r'^RT[\s]+'
combined_pat = r'|'.join((pat1, pat2, pat3))


def get_analysis_indo(score):
    if score == 0:
        return 'Negative'
    else:
        return 'Positive'


def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()


def visualize_wordcloud(text, stop_words, language):
    wordcloud = WordCloud(width=3000, height=2000,
                          max_words=200, colormap='Set3',
                          background_color="black",
                          stopwords=stop_words).generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig("WC_" + language + ".jpg")
    img = Image.open("WC_" + language + ".jpg")
    return img


def visualize_sentiment_indo(df, language):
    plt.figure(figsize=(15, 10), facecolor='k')
    plt.title('Sentiment Analysis', fontsize=40, pad=20)
    plt.xlabel('Sentiment', fontsize=30, labelpad=20)
    plt.ylabel('Count', fontsize=30, labelpad=20)
    sns.countplot(x=df,
                  data=df,
                  palette=["Green", "Red"])
    plt.savefig("Sentiment_" + language + ".jpg")
    img = Image.open("Sentiment_" + language + ".jpg")
    return img
