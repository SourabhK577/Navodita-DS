import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def download_and_extract_data(data_dir="data"):
    """
    Downloads and extracts the MovieLens 100k latest small dataset if not already present.
    Has self-healing capabilities if a corrupted zip file is detected.
    """
    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    extract_path = os.path.join(data_dir, "ml-latest-small")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    # Check if extraction exists, if not proceed to download/extract
    if not os.path.exists(extract_path):
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100:  # 100 bytes is too small for a valid zip
            print(f"Downloading MovieLens dataset from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")
            
        print("Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print("Warning: Corrupted zip file detected. Deleting and re-downloading...")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            # Re-download
            urllib.request.urlretrieve(url, zip_path)
            # Re-extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Re-download and extraction complete.")
            
    return os.path.join(data_dir, "ml-latest-small")


class MovieRecommender:
    """
    Simple Movie Recommendation Engine using Collaborative Filtering and Genre Similarity.
    """
    def __init__(self, data_path):
        self.movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
        self.ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
        
        # Clean title spaces and parse release year
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')
        
        # Merge for convenient lookups
        self.ratings_merged = pd.merge(self.ratings, self.movies, on="movieId")
        
        # Create user-movie matrix
        self.user_movie_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating')
        self.user_movie_matrix_filled = self.user_movie_matrix.fillna(0)
        
        # Precompute user similarity matrix
        self.user_similarity = cosine_similarity(self.user_movie_matrix_filled)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )
        
        # Precompute genre similarity matrix
        self.genres_df = self.movies['genres'].str.get_dummies(sep='|')
        self.genre_similarity = cosine_similarity(self.genres_df)
        self.genre_similarity_df = pd.DataFrame(
            self.genre_similarity,
            index=self.movies['movieId'],
            columns=self.movies['movieId']
        )

    def get_popular_movies(self, num_movies=10):
        """
        Returns top popular movies based on number of ratings and average rating.
        """
        stats = self.ratings.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        )
        # Filter to movies with at least 50 ratings
        popular = stats[stats['rating_count'] >= 50]
        popular_ids = popular.sort_values(by='avg_rating', ascending=False).head(num_movies).index
        
        result = self.movies[self.movies['movieId'].isin(popular_ids)].copy()
        result = pd.merge(result, popular, on='movieId')
        return result.sort_values(by='avg_rating', ascending=False)

    def get_user_recommendations(self, user_id, num_recommendations=10):
        """
        User-Based Collaborative Filtering recommendations.
        Predicts ratings for unrated movies as the weighted average rating from similar users.
        """
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame()
            
        user_ratings = self.user_movie_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings.isna()].index
        
        # Cold start check: if user has rated nothing, show popular movies
        if user_ratings.notna().sum() == 0:
            return self.get_popular_movies(num_recommendations)
            
        # Get user similarity scores sorted
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)
        similar_users = similar_users.drop(user_id) # exclude current user
        
        # Take top 30 similar users
        top_users = similar_users.head(30)
        
        predictions = {}
        for movie_id in unrated_movies:
            # Ratings from similar users for this movie
            movie_ratings = self.user_movie_matrix.loc[top_users.index, movie_id]
            valid_ratings = movie_ratings.dropna()
            
            if len(valid_ratings) == 0:
                continue
                
            # Get similarity scores for users who rated this movie
            sim_scores = top_users.loc[valid_ratings.index]
            
            weighted_sum = (valid_ratings * sim_scores).sum()
            sim_sum = sim_scores.sum()
            
            if sim_sum > 0:
                predictions[movie_id] = weighted_sum / sim_sum
                
        if not predictions:
            return self.get_popular_movies(num_recommendations)
            
        # Sort predictions
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        rec_ids = [x[0] for x in sorted_predictions]
        rec_ratings = [x[1] for x in sorted_predictions]
        
        recs_df = self.movies[self.movies['movieId'].isin(rec_ids)].copy()
        recs_df['predicted_rating'] = recs_df['movieId'].map(dict(sorted_predictions))
        return recs_df.sort_values(by='predicted_rating', ascending=False)

    def get_custom_user_recommendations(self, custom_ratings, num_recommendations=10):
        """
        Calculates User-Based Collaborative Filtering recommendations for a temporary user profile.
        custom_ratings: dict mapping movieId (int) to rating (float)
        """
        # Create a ratings series with the same columns as the user-movie matrix
        new_user_ratings = pd.Series(index=self.user_movie_matrix.columns, dtype=float)
        for movie_id, rating in custom_ratings.items():
            new_user_ratings.loc[movie_id] = rating
            
        if new_user_ratings.notna().sum() == 0:
            return self.get_popular_movies(num_recommendations)
            
        # Calculate similarity between new user and existing users
        new_user_filled = new_user_ratings.fillna(0).values.reshape(1, -1)
        sim_scores = cosine_similarity(new_user_filled, self.user_movie_matrix_filled)[0]
        sim_series = pd.Series(sim_scores, index=self.user_movie_matrix.index)
        
        # Get top 30 similar users (excluding any complete cold-start 0 similarities if possible)
        top_users = sim_series.sort_values(ascending=False).head(30)
        
        # Movies the new user hasn't rated
        unrated_movies = new_user_ratings[new_user_ratings.isna()].index
        
        predictions = {}
        for movie_id in unrated_movies:
            movie_ratings = self.user_movie_matrix.loc[top_users.index, movie_id]
            valid_ratings = movie_ratings.dropna()
            
            if len(valid_ratings) == 0:
                continue
                
            sims = top_users.loc[valid_ratings.index]
            weighted_sum = (valid_ratings * sims).sum()
            sim_sum = sims.sum()
            
            if sim_sum > 0:
                predictions[movie_id] = weighted_sum / sim_sum
                
        if not predictions:
            return self.get_popular_movies(num_recommendations)
            
        # Sort predictions and fetch top movies
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        rec_ids = [x[0] for x in sorted_predictions]
        
        recs_df = self.movies[self.movies['movieId'].isin(rec_ids)].copy()
        recs_df['predicted_rating'] = recs_df['movieId'].map(dict(sorted_predictions))
        return recs_df.sort_values(by='predicted_rating', ascending=False)

    def get_similar_movies_by_ratings(self, movie_id, num_recommendations=10):
        """
        Item-Based Collaborative Filtering (similarity by rating behavior).
        """
        # Build rating-based movie-user matrix
        movie_user_matrix = self.user_movie_matrix.T.fillna(0)
        movie_sim = cosine_similarity(movie_user_matrix)
        movie_sim_df = pd.DataFrame(
            movie_sim,
            index=movie_user_matrix.index,
            columns=movie_user_matrix.index
        )
        
        if movie_id not in movie_sim_df.index:
            return pd.DataFrame()
            
        similar_scores = movie_sim_df[movie_id].sort_values(ascending=False).drop(movie_id)
        similar_ids = similar_scores.head(num_recommendations).index
        
        recs_df = self.movies[self.movies['movieId'].isin(similar_ids)].copy()
        recs_df['similarity_score'] = recs_df['movieId'].map(similar_scores)
        return recs_df.sort_values(by='similarity_score', ascending=False)

    def get_similar_movies_by_genres(self, movie_id, num_recommendations=10):
        """
        Content-Based Similarity (similarity by genres).
        """
        if movie_id not in self.genre_similarity_df.index:
            return pd.DataFrame()
            
        similar_scores = self.genre_similarity_df[movie_id].sort_values(ascending=False).drop(movie_id)
        similar_ids = similar_scores.head(num_recommendations).index
        
        recs_df = self.movies[self.movies['movieId'].isin(similar_ids)].copy()
        recs_df['similarity_score'] = recs_df['movieId'].map(similar_scores)
        return recs_df.sort_values(by='similarity_score', ascending=False)

if __name__ == "__main__":
    # Test data download and model initialization
    path = download_and_extract_data()
    print("Loading recommender model...")
    recommender = MovieRecommender(path)
    print("Popular movies:")
    print(recommender.get_popular_movies(5)[['title', 'genres']])
    
    print("\nRecommendations for User ID 1:")
    print(recommender.get_user_recommendations(1, 5)[['title', 'predicted_rating']])
