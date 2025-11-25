import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


class PCARecommender:
    def __init__(self, cinema, n_components=None):
        self.cinema = cinema
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None

        self.feature_matrix = None
        self.feature_names = None
        self.movie_ids = None

        self.X_scaled = None
        self.X_pca = None
        self.reconstructed = None
        self.explained_variance_ratio_ = None
        self.components_ = None

    def create_movie_feature_matrix(self):
        """Создание семантически осмысленной матрицы признаков фильмов"""
        movies = self.cinema.movies.copy()

        if "year" not in movies.columns:
            movies["year"] = pd.NA
        movies["year"] = movies["year"].fillna(movies["year"].median())

        all_genres = set()
        for genres_list in movies["genres"]:
            if isinstance(genres_list, list):
                all_genres.update(genres_list)

        all_genres = sorted(list(all_genres))

        for genre in all_genres:
            movies[f"genre_{genre}"] = movies["genres"].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )

        current_year = datetime.now().year
        movies["movie_age"] = (current_year - movies["year"]).fillna(0).astype(float)
        movies["is_old"] = (movies["movie_age"] > 30).astype(int)
        movies["is_recent"] = (movies["movie_age"] < 10).astype(int)

        if isinstance(self.cinema.ratings.index, pd.MultiIndex):
            ratings_df = self.cinema.ratings.reset_index()
        else:
            ratings_df = self.cinema.ratings.copy().reset_index()

        rating_stats = (
            ratings_df.groupby("movieId")["rating"]
            .agg(["mean", "std", "count"])
            .rename(
                columns={
                    "mean": "rating_mean",
                    "std": "rating_std",
                    "count": "rating_count",
                }
            )
        )
        rating_stats = rating_stats.fillna(0)
        movies = movies.join(rating_stats, how="left")

        movies["rating_mean"] = movies["rating_mean"].fillna(0).astype(float)
        movies["rating_std"] = movies["rating_std"].fillna(0).astype(float)
        movies["rating_count"] = movies["rating_count"].fillna(0).astype(float)

        numeric_features = ["movie_age", "rating_mean", "rating_std", "rating_count"]
        for feature in numeric_features:
            if movies[feature].std(ddof=0) == 0:
                movies[f"{feature}_norm"] = 0.0
            else:
                movies[f"{feature}_norm"] = (
                    movies[feature] - movies[feature].mean()
                ) / movies[feature].std(ddof=0)

        genre_cols = [f"genre_{g}" for g in all_genres]
        feature_cols = genre_cols + [
            "is_old",
            "is_recent",
            "movie_age_norm",
            "rating_mean_norm",
            "rating_std_norm",
            "rating_count_norm",
        ]

        for col in feature_cols:
            if col not in movies.columns:
                movies[col] = 0.0

        self.feature_matrix = movies[feature_cols].fillna(0).values.astype(float)
        self.feature_names = feature_cols
        self.movie_ids = movies.index.tolist()

        return self.feature_matrix

    def build_pca_model(self):
        """Построение PCA модели"""
        try:
            X = self.create_movie_feature_matrix()
            if X is None or X.size == 0:
                raise ValueError("Пустая матрица признаков фильмов")

            self.X_scaled = self.scaler.fit_transform(X)

            self.pca = PCA(n_components=self.n_components)
            self.X_pca = self.pca.fit_transform(self.X_scaled)

            X_reconstructed_scaled = self.pca.inverse_transform(self.X_pca)
            self.reconstructed = self.scaler.inverse_transform(X_reconstructed_scaled)

            self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
            self.components_ = self.pca.components_

            return True
        except Exception as e:
            print(f"Ошибка построения PCA: {e}")
            return False

    def evaluate_reconstruction(self):
        """Оценка качества восстановления"""
        if self.reconstructed is None:
            return None, None, None

        X_original = self.create_movie_feature_matrix()
        mse = mean_squared_error(X_original.flatten(), self.reconstructed.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(X_original.flatten(), self.reconstructed.flatten())
        return mse, rmse, mae

    def recommend_similar_movies(self, movie_id, top_k=10):
        """Рекомендация похожих фильмов"""
        if self.pca is None or self.X_pca is None:
            if not self.build_pca_model():
                return []

        try:
            idx = self.movie_ids.index(movie_id)
        except ValueError:
            return []

        target_vec = self.X_pca[idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.X_pca)[0]
        pairs = [
            (mid, float(sim))
            for mid, sim in zip(self.movie_ids, sims)
            if mid != movie_id
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def recommend_for_current_user(self, top_k=20):
        """Рекомендации для текущего пользователя на основе PCA"""
        if self.pca is None or self.X_pca is None:
            if not self.build_pca_model():
                return pd.DataFrame(columns=["movieId", "score"])

        rated = self.cinema.rates.copy()
        if rated.empty:
            return pd.DataFrame(columns=["movieId", "score"])

        movie_idx_map = {mid: i for i, mid in enumerate(self.movie_ids)}
        user_vec = None
        weights_sum = 0.0

        for mid, row in rated.iterrows():
            if mid not in movie_idx_map:
                continue
            idx = movie_idx_map[mid]
            rating = float(row["rating"])
            vec = self.X_pca[idx]
            if user_vec is None:
                user_vec = rating * vec
            else:
                user_vec += rating * vec
            weights_sum += abs(rating)

        if user_vec is None or weights_sum == 0:
            return pd.DataFrame(columns=["movieId", "score"])

        user_vec = (user_vec / weights_sum).reshape(1, -1)

        sims = cosine_similarity(user_vec, self.X_pca)[0]
        results = []
        for mid, sim in zip(self.movie_ids, sims):
            if mid in rated.index:
                continue
            results.append((mid, float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]

        df = pd.DataFrame(top, columns=["movieId", "score"])
        movies_meta = self.cinema.movies.loc[df["movieId"]].reset_index()
        df = df.merge(
            movies_meta[["movieId", "title", "genres"]], on="movieId", how="left"
        )

        return df

    def get_component_analysis(self, component_idx, top_n=20):
        """Анализ компоненты"""
        if self.pca is None or self.components_ is None:
            return None

        if component_idx < 0 or component_idx >= self.components_.shape[0]:
            return None

        comp = self.components_[component_idx]
        feat_importance = list(zip(self.feature_names, comp))
        feat_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        return feat_importance[:top_n]

    def plot_variance_explained(self):
        """График объясненной дисперсии"""
        if self.pca is None or self.explained_variance_ratio_ is None:
            print("PCA модель ещё не построена")
            return

        cum_var = np.cumsum(self.explained_variance_ratio_)
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker="o")
        plt.xlabel("Число компонент")
        plt.ylabel("Объясненная дисперсия")
        plt.title("PCA: объясненная дисперсия")
        plt.grid(True)
        plt.show()

    def cluster_movies(self, n_clusters=6, method="agglomerative"):
        """Кластеризация фильмов в PCA-пространстве"""
        if self.pca is None or self.X_pca is None:
            if not self.build_pca_model():
                return None

        if method == "gmm":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(self.X_pca)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(self.X_pca)

        plt.figure(figsize=(8, 6))
        if self.X_pca.shape[1] >= 2:
            plt.scatter(
                self.X_pca[:, 0], self.X_pca[:, 1], c=labels, cmap="tab10", s=10
            )
            plt.title(f"Movie clusters (method={method})")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True)
            plt.show()

        return labels

    def get_recommendation_explanation(self, movie_id, top_features=5):
        """Объяснение рекомендации для фильма"""
        if self.pca is None:
            return "Модель PCA не построена"

        try:
            movie_idx = self.movie_ids.index(movie_id)
            movie_features = self.feature_matrix[movie_idx]

            feature_importance = []
            for i, (name, value) in enumerate(zip(self.feature_names, movie_features)):
                if value != 0:
                    feature_importance.append((name, value))

            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            explanation = f"Фильм характеризуется: "
            features_desc = []
            for name, value in feature_importance[:top_features]:
                if "genre_" in name:
                    genre = name.replace("genre_", "")
                    features_desc.append(f"жанр '{genre}'")
                elif "rating_mean" in name:
                    features_desc.append(
                        f"высоким средним рейтингом"
                        if value > 0
                        else f"низким средним рейтингом"
                    )
                elif "is_old" in name and value > 0:
                    features_desc.append("старым релизом")
                elif "is_recent" in name and value > 0:
                    features_desc.append("новым релизом")

            explanation += ", ".join(features_desc)
            return explanation

        except ValueError:
            return "Фильм не найден в базе"
