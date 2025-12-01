import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os

# ---------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("### Find similar movies using Machine Learning (TF-IDF + Cosine Similarity)")


# ---------------------------------------------------------------------
# FUNCTION TO LOAD MOVIES + CREDITS (FROM SAME FOLDER)
# ---------------------------------------------------------------------
def load_csv_safely(filename):
    """
    This function will load:
    - CSV directly
    - OR CSV inside ZIP
    """
    if os.path.exists(filename):
        if filename.endswith(".zip"):
            with zipfile.ZipFile(filename) as z:
                for file in z.namelist():
                    if file.endswith(".csv"):
                        return pd.read_csv(z.open(file))
        else:
            return pd.read_csv(filename)
    return None


@st.cache_data
def load_dataset():
    st.info("Loading movie dataset... Please wait.")

    # LOAD MOVIES CSV
    movies = load_csv_safely("tmdb_5000_movies.csv")
    if movies is None:
        movies = load_csv_safely("tmdb_5000_movies.csv.zip")

    # LOAD CREDITS CSV
    credits = load_csv_safely("tmdb_5000_credits.csv")
    if credits is None:
        credits = load_csv_safely("tmdb_5000_credits.csv.zip")

    if movies is None:
        st.error("❌ ERROR: tmdb_5000_movies.csv NOT FOUND in this folder.")
        st.stop()

    if credits is None:
        st.warning("⚠ Credits file missing. Recommendations will still work but with less info.")
    else:
        credits.rename(columns={"movie_id": "id"}, inplace=True)
        movies = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    return movies


movies = load_dataset()

# Rename title if necessary
if "title" not in movies.columns and "original_title" in movies.columns:
    movies.rename(columns={"original_title": "title"}, inplace=True)


# ---------------------------------------------------------------------
# BUILD RECOMMENDER SYSTEM (TF-IDF)
# ---------------------------------------------------------------------
@st.cache_data
def build_model():
    st.info("Building TF-IDF Matrix...")
    movies["overview"] = movies["overview"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["overview"])

    st.success("TF-IDF Matrix Ready!")
    return tfidf_matrix


tfidf_matrix = build_model()
cosine_sim = cosine_similarity(tfidf_matrix)


# ---------------------------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------------------------
def recommend(title, top_n=10):
    if title not in movies["title"].values:
        return None

    idx = movies[movies["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices][["title", "overview", "vote_average", "release_date"]]


# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------
st.sidebar.header("🔍 Search Movie")

movie_list = sorted(movies["title"].dropna().unique())

selected_movie = st.sidebar.selectbox("Choose a movie:", movie_list)

number = st.sidebar.slider("Number of recommendations:", 5, 20, 10)

if st.sidebar.button("Recommend"):
    results = recommend(selected_movie, number)

    if results is None:
        st.error("❌ Movie not found!")
    else:
        st.success(f"🎯 Showing top {number} movies similar to **{selected_movie}**")

        for index, row in results.iterrows():
            st.markdown("---")
            st.markdown(f"### 🎞 {row['title']}")
            st.markdown(f"⭐ Rating: **{row['vote_average']}**")
            st.markdown(f"📅 Release: **{str(row['release_date'])[:4]}**")
            st.write(row["overview"])
