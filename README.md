# 📈 Netflix Content Strategy Insights Project

This project explores Netflix’s content data to extract actionable insights using data cleaning, processing, visualization, clustering, and sentiment analysis – all developed in a Google Colab notebook. It aims to support strategic decisions in content planning, release timing, and regional focus.
🚀 Live App: [Click here to launch the app] (https://netflix-content-analysis-xuox47rlk8wuzyl77frota.streamlit.app/)

---

## 📁 Dataset

- **Source**: Netflix Movies and TV Shows Dataset ([Kaggle Link](https://www.kaggle.com/datasets/shivamb/netflix-shows))
- **File Used**: `netflix_data_cleaned.csv` (cleaned and processed)

---

## 🔍 Key Analysis Performed

1. **Data Cleaning & Preprocessing**
   - Null value handling
   - Date formatting (`date_added`)
   - String operations and splitting genres/countries

2. **Content Type Distribution**
   - Movies vs TV Shows count

3. **Yearly Content Additions**
   - Bar graph showing trend of content added each year

4. **Top 10 Countries by Content**
   - Extracted from `country` column with proper parsing

5. **Popular Genres**
   - Genre frequency analysis

6. **Content Rating Distribution**
   - Analyzing content maturity and audience targeting

7. **Best Months to Release Content**
   - Month-wise content addition trend

8. **Clustering by Genre**
   - Used KMeans to cluster shows based on genre data

9. **Sentiment Analysis on Descriptions**
   - Used `TextBlob` for polarity scoring of show descriptions

10. **Rating vs Sentiment Correlation**
    - Scatter plot showing relationship

11. **Top Actors**
    - Most frequently appearing cast members

12. **Time Series Analysis**
    - Monthly content addition trend visualization

---

## 🧪 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `plotly`
- `wordcloud`
- `textblob`
- `sklearn` (for clustering)
- `datetime`

---

## 💻 How to Run

1. Open the `Netflix_Colab_Insights.ipynb` notebook in [Google Colab](https://colab.research.google.com/)
2. Upload the `netflix_data_cleaned.csv` file
3. Run cells sequentially for insights and visualizations

---

## 📸 Visual Output Samples

> Inside screenshots folder..

---

## ✨ Author

Aryan Sharma 
B.Tech CSE (AI & ML) | Data Enthusiast  
📧 aryan.sharma6428@gmail.com 
🔗 www.linkedin.com/aryan-sharma-3914aa255

---

## 📄 License

Open-source under the MIT License.

