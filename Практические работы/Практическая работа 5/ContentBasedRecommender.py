import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ContentBasedRecommender:
    def __init__(self, cinema, n_clusters=12, random_state=42):
        self.cinema = cinema
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.feature_matrix = None
        self.feature_df = None
        self.cluster_labels = None
        self.kmeans = None

    def build_feature_matrix(self):
        movies = self.cinema.movies.copy()

        genres_df = movies["genres"].apply(lambda g: " ".join(g) if isinstance(g, list) else "")
        genres_ohe = genres_df.str.get_dummies(sep=" ")

        tags = self.cinema.tags.reset_index()

        tags["tag"] = tags["tag"].fillna("")
        tags_text = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x))

        movies["tags_text"] = movies.index.map(tags_text).fillna("")

        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=500)
        tags_tfidf = tfidf.fit_transform(movies["tags_text"]).toarray()

        tags_tfidf_df = pd.DataFrame(tags_tfidf,
                                    index=movies.index,
                                    columns=[f"tag_{i}" for i in range(tags_tfidf.shape[1])])

        ratings = self.cinema.ratings.reset_index()

        avg_rating = ratings.groupby("movieId")["rating"].mean()
        rating_count = ratings.groupby("movieId")["rating"].count()

        movies["avg_rating"] = movies.index.map(avg_rating)
        movies["rating_count"] = movies.index.map(rating_count)

        movies["avg_rating"] = movies["avg_rating"].fillna(movies["avg_rating"].mean())
        movies["rating_count"] = movies["rating_count"].fillna(0)

        movies["year_filled"] = movies["year"].fillna(movies["year"].median())

        num_df = movies[["avg_rating", "rating_count", "year_filled"]]
        num_df = num_df.fillna(0)

        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_df)

        num_scaled_df = pd.DataFrame(num_scaled,
                                    index=movies.index,
                                    columns=["avg_rating_scaled", "rating_count_scaled", "year_scaled"])

        self.feature_df = pd.concat([genres_ohe, tags_tfidf_df, num_scaled_df], axis=1)

        self.feature_df = self.feature_df.fillna(0)

        self.feature_matrix = self.feature_df.values

        return True

    def cluster_content(self):
        if self.feature_matrix is None:
            self.build_feature_matrix()

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = self.kmeans.fit_predict(self.feature_matrix)

        self.cluster_labels = pd.Series(labels, index=self.feature_df.index, name="cluster")

        return self.cluster_labels

    def recommend_from_cluster(self, movie_id, top_k=15):
        """
        Находит фильмы из того же контентного кластера.
        """
        if self.cluster_labels is None:
            self.cluster_content()

        if movie_id not in self.cluster_labels.index:
            return pd.DataFrame()

        movie_cluster = self.cluster_labels[movie_id]

        same_cluster = self.cluster_labels[self.cluster_labels == movie_cluster].index
        same_cluster = same_cluster[same_cluster != movie_id]

        df = self.cinema.movies.loc[same_cluster].copy()
        df["cluster"] = movie_cluster

        return df.head(top_k)

    def recommend_for_user(self, top_k=20):
        """
        Пользователь оценивает фильмы → определяем его любимые кластеры.
        """
        if self.cluster_labels is None:
            self.cluster_content()

        rated = self.cinema.rates.index.tolist()
        if not rated:
            return pd.DataFrame()

        user_clusters = self.cluster_labels.loc[rated].value_counts()

        if user_clusters.empty:
            return pd.DataFrame()

        fav_cluster = user_clusters.index[0]

        movies_in_cluster = self.cluster_labels[self.cluster_labels == fav_cluster].index
        unseen = [m for m in movies_in_cluster if m not in rated]

        df = self.cinema.movies.loc[unseen].copy()
        df["cluster"] = fav_cluster

        return df.head(top_k)
