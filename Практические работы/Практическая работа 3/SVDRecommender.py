import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering


class SVDRecommender:
    def __init__(self, cinema, k=20):
        self.cinema = cinema
        self.k = k
        self.U = None
        self.D = None
        self.Vt = None
        self.pred_matrix = None

    def build(self):
        """SVD по матрице рейтингов"""
        R = self.cinema.get_rating_matrix()

        R_filled = R.fillna(0).values

        U, D, Vt = np.linalg.svd(R_filled, full_matrices=False)

        U_k = U[:, : self.k]
        D_k = np.diag(D[: self.k])
        Vt_k = Vt[: self.k, :]

        self.U = U_k
        self.D = D_k
        self.Vt = Vt_k

        self.pred_matrix = np.dot(U_k, np.dot(D_k, Vt_k))

        return self.pred_matrix

    def evaluate(self):
        """Вычисление RMSE, MSE, MAE по реальным оценкам"""
        ratings = self.cinema.ratings.reset_index()
        movie_ids = self.cinema.get_rating_matrix().columns.tolist()
        user_ids = self.cinema.get_rating_matrix().index.tolist()

        preds = []
        reals = []

        for _, row in ratings.iterrows():
            uid = row["userId"]
            mid = row["movieId"]

            if uid not in user_ids or mid not in movie_ids:
                continue

            u_idx = user_ids.index(uid)
            m_idx = movie_ids.index(mid)
            pred = self.pred_matrix[u_idx, m_idx]

            preds.append(pred)
            reals.append(row["rating"])

        mse = mean_squared_error(reals, preds)
        rmse = root_mean_squared_error(reals, preds)
        mae = mean_absolute_error(reals, preds)

        return mse, rmse, mae

    def recommend_for_current_user(self, top_k=20):
        """Рекомендации для текущего пользователя"""
        R = self.cinema.get_rating_matrix()
        movie_ids = R.columns.tolist()

        current_user_id = -1
        current_ratings = pd.Series([np.nan] * len(movie_ids), index=movie_ids)

        for mid in self.cinema.rates.index:
            if mid in movie_ids:
                current_ratings[mid] = self.cinema.rates.loc[mid, "rating"]

        R_ext = pd.concat(
            [R, pd.DataFrame([current_ratings], index=[current_user_id])]
        ).fillna(0)

        u_vec = np.dot(
            np.dot(R_ext.loc[current_user_id].values, self.Vt.T), np.linalg.inv(self.D)
        )

        preds = np.dot(u_vec, self.Vt)

        result = pd.DataFrame({"movieId": movie_ids, "pred": preds})

        result = result[~result["movieId"].isin(self.cinema.rates.index)]

        result = result.sort_values("pred", ascending=False).head(top_k)

        return self.cinema.movies.loc[result["movieId"]]

    def evaluate_over_k(self, k_min=2, k_max=50):
        R = self.cinema.get_rating_matrix()
        R_filled = R.fillna(0).values
        ratings = self.cinema.ratings.reset_index()

        movie_ids = R.columns.tolist()
        user_ids = R.index.tolist()

        ks = []
        mses = []
        rmses = []
        maes = []

        for k in range(k_min, k_max + 1):
            self.cinema.console.print(f"Построение SVD для k={k}: ", highlight=False, end="")

            U, D, Vt = np.linalg.svd(R_filled, full_matrices=False)

            U_k = U[:, :k]
            D_k = np.diag(D[:k])
            Vt_k = Vt[:k, :]

            pred_matrix = np.dot(U_k, np.dot(D_k, Vt_k))

            preds = []
            reals = []

            for _, row in ratings.iterrows():
                uid = row["userId"]
                mid = row["movieId"]

                if uid not in user_ids or mid not in movie_ids:
                    continue

                u_idx = user_ids.index(uid)
                m_idx = movie_ids.index(mid)

                preds.append(pred_matrix[u_idx, m_idx])
                reals.append(row["rating"])

            mse = mean_squared_error(reals, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(reals, preds)

            self.cinema.console.print(f"[green]MSE:[/green] {mse:.4f}, [green]RMSE:[/green] {rmse:.4f}, [green]MAE:[/green] {mae:.4f}")

            ks.append(k)
            mses.append(mse)
            rmses.append(rmse)
            maes.append(mae)

        plt.figure(figsize=(10, 6))
        plt.plot(ks, mses, label="MSE")
        plt.plot(ks, rmses, label="RMSE")
        plt.plot(ks, maes, label="MAE")
        plt.xlabel("Число скрытых факторов")
        plt.ylabel("Значение метрики")
        plt.title("Зависимость MSE, RMSE, MAE от числа факторов k")
        plt.legend()
        plt.grid(True)
        plt.show()

    def cluster_users(self, n_clusters=4):
        if self.U is None:
            raise ValueError("Сначала постройте SVD с помощью build().")

        U_k = self.U

        pca = PCA(n_components=2)
        U_2d = pca.fit_transform(U_k)

        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(U_k)

        plt.figure(figsize=(8, 6))
        plt.scatter(U_2d[:, 0], U_2d[:, 1], c=labels, cmap="tab10")
        plt.title(f"Кластеризация пользователей")
        plt.xlabel("PCA component 1")
        plt.ylabel("PCA component 2")
        plt.grid(True)
        plt.show()

        return labels
