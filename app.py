import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from recommender import download_and_extract_data, MovieRecommender

# Set page configuration
st.set_page_config(
    page_title="MovieRecom - Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os

# Load external CSS styling file
def local_css(file_name):
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

# Load data and initialize recommender (cached for fast performance)
@st.cache_resource
def get_recommender():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")
    path = download_and_extract_data(data_path)
    return MovieRecommender(path)

with st.spinner("Initializing Recommendation Engine..."):
    recommender = get_recommender()

# Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🎬 MovieRecom</h1>
    <p class="hero-subtitle">Interactive Movie Recommendation Engine using User-Based Collaborative Filtering & Genre Similarity</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=300&auto=format&fit=crop", use_container_width=True)
st.sidebar.markdown("### ⚙️ Engine Settings")

input_mode = st.sidebar.radio(
    "🎯 Recommendation Mode",
    ["🍿 Custom Profile (Select Movies)", "👤 Preset User Profile"]
)

movie_titles = recommender.movies['title'].sort_values().tolist()

if input_mode == "👤 Preset User Profile":
    user_ids = sorted(recommender.ratings['userId'].unique())
    selected_user = st.sidebar.selectbox("Select Target User ID", user_ids, index=0)
    user_ratings = recommender.ratings[recommender.ratings['userId'] == selected_user]
    st.sidebar.markdown(f"**Ratings Submitted:** `{len(user_ratings)}`")
    st.sidebar.markdown(f"**Average Rating:** `{user_ratings['rating'].mean():.2f} ★`")
else:
    st.sidebar.markdown("### 🔍 Movies You Love")
    st.sidebar.markdown("Search and select up to 5 movies you love. We'll give you recommendations based on these choices!")
    
    # Pre-select some default popular movies so the app looks ready out-of-the-box
    default_selections = []
    for default_title in ["Toy Story (1995)", "Matrix, The (1999)", "Pulp Fiction (1994)", "Forrest Gump (1994)", "Shawshank Redemption, The (1994)"]:
        if default_title in movie_titles:
            default_selections.append(default_title)
            
    chosen_movies = st.sidebar.multiselect(
        "Select up to 5 movies:",
        movie_titles,
        default=default_selections[:5],
        max_selections=5
    )

# ================= PERSONALIZED RECOMMENDATIONS =================
st.markdown("### 🍿 Recommended Movies for You")

if input_mode == "🍿 Custom Profile (Select Movies)":
    st.markdown("These suggestions are generated via **User-Based Collaborative Filtering** by finding other users who also highly rated your favorite movies.")
    
    num_recs = st.slider("Number of recommendations to generate", 3, 20, 8, key="num_recs_slider_custom")
    
    if len(chosen_movies) == 0:
        st.warning("⚠️ Please select at least one movie in the sidebar to generate recommendations.")
    else:
        if st.button("🚀 Generate Recommendations", type="primary"):
            # Construct custom ratings dictionary
            # Give each selected movie a 5.0 (highly loved) rating
            custom_ratings = {}
            for title in chosen_movies:
                m_row = recommender.movies[recommender.movies['title'] == title]
                if not m_row.empty:
                    m_id = m_row.iloc[0]['movieId']
                    custom_ratings[m_id] = 5.0
                
            with st.spinner("Analyzing ratings and similarity vectors..."):
                recs = recommender.get_custom_user_recommendations(custom_ratings, num_recs)
                
            if recs.empty:
                st.warning("No custom recommendations could be generated. Showing popular movies.")
                recs = recommender.get_popular_movies(num_recs)
                
            # Display recommendations in grid
            cols = st.columns(4)
            for idx, row in recs.reset_index(drop=True).iterrows():
                col_idx = idx % 4
                genres_list = row['genres'].split('|')
                genres_html = "".join([f'<span class="movie-genre-badge">{g}</span>' for g in genres_list])
                
                # Predict stars
                pred_rating = row.get('predicted_rating', 0.0)
                star_rating = "★" * int(round(pred_rating)) + "☆" * (5 - int(round(pred_rating)))
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div>
                            <div class="movie-title">{row['title']}</div>
                            <div class="movie-genres-container">{genres_html}</div>
                        </div>
                        <div class="rating-container">
                            <span class="rating-stars">{star_rating}</span>
                            <span class="rating-val">{pred_rating:.2f}/5</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.markdown("These suggestions are generated via **User-Based Collaborative Filtering** by analyzing users with similar rating patterns to you.")
    
    num_recs = st.slider("Number of recommendations to generate", 3, 20, 8, key="num_recs_slider_preset")
    
    if st.button("🚀 Generate Recommendations", type="primary"):
        with st.spinner("Analyzing similarities..."):
            recs = recommender.get_user_recommendations(selected_user, num_recs)
            
        if recs.empty:
            st.warning("No recommendations found. Showing popular movies instead.")
            recs = recommender.get_popular_movies(num_recs)
            
        # Display recommendations in grid
        cols = st.columns(4)
        for idx, row in recs.reset_index(drop=True).iterrows():
            col_idx = idx % 4
            genres_list = row['genres'].split('|')
            genres_html = "".join([f'<span class="movie-genre-badge">{g}</span>' for g in genres_list])
            
            # Predict stars
            pred_rating = row.get('predicted_rating', 0.0)
            star_rating = "★" * int(round(pred_rating)) + "☆" * (5 - int(round(pred_rating)))
            
            with cols[col_idx]:
                st.markdown(f"""
                <div class="movie-card">
                    <div>
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-genres-container">{genres_html}</div>
                    </div>
                    <div class="rating-container">
                        <span class="rating-stars">{star_rating}</span>
                        <span class="rating-val">{pred_rating:.2f}/5</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown("---")
    st.markdown("#### 📜 Movies Already Rated by this User")
    # Show user's highest rated movies
    user_top_movies = user_ratings.merge(recommender.movies, on='movieId').sort_values(by='rating', ascending=False).head(10)
    st.dataframe(
        user_top_movies[['title', 'genres', 'rating']].rename(
            columns={'title': 'Movie Title', 'genres': 'Genres', 'rating': 'User Rating'}
        ),
        use_container_width=True,
        hide_index=True
    )
