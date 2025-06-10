import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import numpy as np

st.set_page_config(page_title="Netflix Dashboard", layout="wide")

# Load data
df = pd.read_csv("netflix_data_cleaned.csv")
df['date_added'] = pd.to_datetime(df['date_added'].astype(str).str.strip(), errors='coerce')
df.dropna(subset=['date_added'], inplace=True)

# Sidebar filters
st.sidebar.header("Filters")
content_types = st.sidebar.multiselect("Select Content Type", df['type'].unique(), default=list(df['type'].unique()))
df = df[df['type'].isin(content_types)]

# Title
st.title("📺 Netflix Content Analysis Dashboard")

# --- 1. Content Type Distribution ---
st.subheader("🎞️ Content Type Distribution")
type_counts = df['type'].value_counts()
st.bar_chart(type_counts)

# --- 2. Content Added Per Year ---
st.subheader("📅 Content Added Per Year")
df['year_added'] = df['date_added'].dt.year
yearly_counts = df['year_added'].value_counts().sort_index()
st.line_chart(yearly_counts)

# --- 3. Top 10 Countries with Most Content ---
st.subheader("🌍 Top 10 Countries with Most Content")
top_countries = df['country'].dropna().str.split(', ').explode().value_counts().head(10)
st.bar_chart(top_countries)

# --- 4. Most Popular Genres ---
st.subheader("🎭 Most Popular Genres")
top_genres = df['listed_in'].str.split(', ').explode().value_counts().head(10)
st.bar_chart(top_genres)

# --- 5. Content Rating Distribution ---
st.subheader("🔞 Content Rating Distribution")
rating_counts = df['rating'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax)
ax.set_ylabel("Number of Titles")
st.pyplot(fig)

# --- 6. Best Months to Release Content ---
st.subheader("🗓️ Best Months to Release Content")
df['month_added'] = df['date_added'].dt.month_name()
month_counts = df['month_added'].value_counts().reindex([
    'January','February','March','April','May','June','July',
    'August','September','October','November','December'
])
fig, ax = plt.subplots()
month_counts.plot(kind='bar', ax=ax)
ax.set_ylabel("Number of Titles")
st.pyplot(fig)

# --- 7. Clustering Titles by Genre ---
st.subheader("🔍 Clustering Titles by Genre")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['listed_in'].fillna(''))
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['genre_cluster'] = kmeans.fit_predict(X)
st.bar_chart(df['genre_cluster'].value_counts())

# --- 8. Sentiment Analysis of Descriptions ---
st.subheader("💬 Sentiment Analysis of Descriptions")
df['description'] = df['description'].fillna("")
df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
fig, ax = plt.subplots()
sns.histplot(df['sentiment'], bins=30, kde=True, ax=ax)
ax.set_title("Sentiment Distribution")
st.pyplot(fig)

# --- 9. Correlation Between Rating and Sentiment ---
st.subheader("📈 Correlation: Rating vs. Sentiment")
rating_sentiment_df = df.dropna(subset=['rating', 'sentiment'])
sentiment_mean = rating_sentiment_df.groupby('rating')['sentiment'].mean()
fig, ax = plt.subplots()
sentiment_mean.plot(kind='bar', ax=ax)
ax.set_ylabel("Average Sentiment Polarity")
st.pyplot(fig)

# --- 10. Top Actors Analysis ---
st.subheader("🎬 Top 10 Most Featured Actors")
top_actors = df['cast'].dropna().str.split(', ').explode().value_counts().head(10)
st.bar_chart(top_actors)

# --- 11. Time Series Analysis ---
st.subheader("📊 Time Series Analysis of Releases")
date_series = df['date_added'].dt.to_period("M").value_counts().sort_index()
date_series.index = date_series.index.to_timestamp()
st.line_chart(date_series)

st.success("✅ Dashboard Loaded Successfully!")


 
