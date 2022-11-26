import module as md
from sklearn.preprocessing import LabelEncoder
import tweepy
from googletrans import Translator
from transformers import pipeline
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_key_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "grey",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

pretrained_indo = "w11wo/indonesian-roberta-base-indolem-sentiment-classifier-fold-0"

nlp_indo = pipeline(
    "sentiment-analysis",
    model=pretrained_indo,
    tokenizer=pretrained_indo
)

user_indo = []
text_indo = []
likes_indo = []
follower_indo = []

for tweet in tweepy.Cursor(api.search_tweets, q='G20 OR "KTT APEC" OR "KTT" OR "KTT G20" -filter:retweets', lang="id").items(100):
    likes_indo.append(tweet.favorite_count)
    text_indo.append(tweet.text)
    user_indo.append(tweet.user.screen_name)
    follower_indo.append(tweet.user.followers_count)

df_indo = pd.DataFrame({'username': user_indo,
                        'tweets': text_indo,
                        'likes': likes_indo,
                        'followers': follower_indo})

# For Indonesia NLP
# New Code
df_indo['sentiment'] = ''
tweet_indo = df_indo['tweets']
data_indo = []

for i in tweet_indo:
    data_indo.append(nlp_indo(i))

predicted_label_indo = []
confidence_indo = []

for my_list in data_indo:
    for item in my_list:
        predicted_label_indo.append(item['label'])
        confidence_indo.append(item['score'])

le = LabelEncoder()
df_indo['sentiment'] = le.fit_transform(predicted_label_indo)


df_indo['label'] = df_indo['sentiment'].apply(md.get_analysis_indo)


def app():
    st.title("G20 Tweets Sentiment Analysis 🔥")
    st.subheader("Analyze the tweets of your favourite Personalities")
    languages = ["Indonesian", "English"]
    choice = st.sidebar.selectbox("Select The Language", languages)

    if choice == "Indonesian":
        analyzer_choice = st.selectbox("Select the Activities",  [
                                       "Generate WordCloud", "Visualize the Sentiment Analysis"])
        if st.button("Analyze"):
            if analyzer_choice == "Generate WordCloud":
                st.success("Generating WordCloud")
                image = md.visualize_wordcloud(text=str(df_indo['tweets'].apply(md.tweet_cleaner)),
                                               stop_words=md.stop_words_indo,
                                               language="Indonesian")
                st.image(image)
            else:
                st.success("Generating the Countplot")
                df_indo['tweets'] = df_indo['tweets'].apply(md.tweet_cleaner)
                image = md.visualize_sentiment_indo(
                    df=df_indo['label'], language="Indonesia")
                st.image(image)


if __name__ == "__main__":
    app()
