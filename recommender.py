import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def download_and_extract_data(data_dir="data"):
    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    extract_path = os.path.join(data_dir, "ml-latest-small")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    if not os.path.exists(extract_path):
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100:
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
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Re-download and extraction complete.")
            
    return os.path.join(data_dir, "ml-latest-small")


class MovieRecommender:
    def __init__(self, data_path):
        self.movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
        self.ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
        
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)')
        self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')
        
        self.user_movie_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating')
        self.user_movie_matrix_filled = self.user_movie_matrix.fillna(0)
        
        self.user_similarity = cosine_similarity(self.user_movie_matrix_filled)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )

    def get_popular_movies(self, num_movies=10):
        stats = self.ratings.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        )
        popular = stats[stats['rating_count'] >= 50]
        popular_ids = popular.sort_values(by='avg_rating', ascending=False).head(num_movies).index
        
        result = self.movies[self.movies['movieId'].isin(popular_ids)].copy()
        result = pd.merge(result, popular, on='movieId')
        return result.sort_values(by='avg_rating', ascending=False)

    def get_user_recommendations(self, user_id, num_recommendations=10):
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame()
            
        user_ratings = self.user_movie_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings.isna()].index
        
        if user_ratings.notna().sum() == 0:
            return self.get_popular_movies(num_recommendations)
            
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)
        similar_users = similar_users.drop(user_id)
        top_users = similar_users.head(30)
        
        predictions = {}
        for movie_id in unrated_movies:
            movie_ratings = self.user_movie_matrix.loc[top_users.index, movie_id]
            valid_ratings = movie_ratings.dropna()
            
            if len(valid_ratings) == 0:
                continue
                
            sim_scores = top_users.loc[valid_ratings.index]
            weighted_sum = (valid_ratings * sim_scores).sum()
            sim_sum = sim_scores.sum()
            
            if sim_sum > 0:
                predictions[movie_id] = weighted_sum / sim_sum
                
        if not predictions:
            return self.get_popular_movies(num_recommendations)
            
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        rec_ids = [x[0] for x in sorted_predictions]
        
        recs_df = self.movies[self.movies['movieId'].isin(rec_ids)].copy()
        recs_df['predicted_rating'] = recs_df['movieId'].map(dict(sorted_predictions))
        return recs_df.sort_values(by='predicted_rating', ascending=False)

    def get_custom_user_recommendations(self, custom_ratings, num_recommendations=10):
        new_user_ratings = pd.Series(index=self.user_movie_matrix.columns, dtype=float)
        for movie_id, rating in custom_ratings.items():
            new_user_ratings.loc[movie_id] = rating
            
        if new_user_ratings.notna().sum() == 0:
            return self.get_popular_movies(num_recommendations)
            
        new_user_filled = new_user_ratings.fillna(0).values.reshape(1, -1)
        sim_scores = cosine_similarity(new_user_filled, self.user_movie_matrix_filled)[0]
        sim_series = pd.Series(sim_scores, index=self.user_movie_matrix.index)
        
        top_users = sim_series.sort_values(ascending=False).head(30)
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
            
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        rec_ids = [x[0] for x in sorted_predictions]
        
        recs_df = self.movies[self.movies['movieId'].isin(rec_ids)].copy()
        recs_df['predicted_rating'] = recs_df['movieId'].map(dict(sorted_predictions))
        return recs_df.sort_values(by='predicted_rating', ascending=False)
