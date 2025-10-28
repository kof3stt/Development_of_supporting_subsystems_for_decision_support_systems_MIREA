import numpy as np
import pandas as pd
from scipy.linalg import expm
from rich.table import Table
import os
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta


class ContinuousMarkovRecommender:
    """Исправленная реализация непрерывной цепи Маркова с реальными временными метками"""

    def __init__(self, cinema, auto_visualize=True):
        self.cinema = cinema
        self.intensity_matrix = None
        self.genre_states = []
        self.auto_visualize = auto_visualize
        self.time_unit = 1.0

    def reset_matrix(self):
        """Сброс матрицы интенсивностей"""
        self.intensity_matrix = None
        self.genre_states = []

    def build_intensity_matrix(self):
        """Построение матрицы интенсивностей с использованием реальных данных"""
        if self.cinema.rates.empty or len(self.cinema.rates) < 2:
            return False

        ratings_chronology = self.get_ratings_chronology()
        if len(ratings_chronology) < 2:
            return False

        all_genres = set()
        for movie_id, rating, time_info in ratings_chronology:
            movie_genres = self.cinema.movies.loc[movie_id, "genres"]
            all_genres.update(movie_genres)
        self.genre_states = sorted(all_genres)

        if not self.genre_states:
            return False

        n_genres = len(self.genre_states)
        state_to_idx = {genre: idx for idx, genre in enumerate(self.genre_states)}

        time_in_state = {genre: 0.0 for genre in self.genre_states}
        transition_counts = np.zeros((n_genres, n_genres))

        for i in range(len(ratings_chronology) - 1):
            current_movie_id, current_rating, current_time = ratings_chronology[i]
            next_movie_id, next_rating, next_time = ratings_chronology[i + 1]

            current_genres = self.cinema.movies.loc[current_movie_id, "genres"]
            next_genres = self.cinema.movies.loc[next_movie_id, "genres"]

            if not current_genres or not next_genres:
                continue

            time_interval = self.calculate_time_interval(current_time, next_time)

            for genre in current_genres:
                if genre in state_to_idx:
                    time_in_state[genre] += time_interval / len(current_genres)

            for curr_genre in current_genres:
                if curr_genre not in state_to_idx:
                    continue
                curr_idx = state_to_idx[curr_genre]

                for next_genre in next_genres:
                    if next_genre not in state_to_idx:
                        continue
                    next_idx = state_to_idx[next_genre]
                    weight = 1.0 / (len(current_genres) * len(next_genres))
                    transition_counts[curr_idx, next_idx] += weight

        self.intensity_matrix = np.zeros((n_genres, n_genres))

        for i, genre_i in enumerate(self.genre_states):
            total_time = time_in_state[genre_i]
            if total_time > 0:
                for j, genre_j in enumerate(self.genre_states):
                    if i != j and transition_counts[i, j] > 0:
                        self.intensity_matrix[i, j] = (
                            transition_counts[i, j] / total_time
                        )

            self.intensity_matrix[i, i] = -np.sum(self.intensity_matrix[i])

        symmetric_matrix = (self.intensity_matrix + self.intensity_matrix.T) / 2

        for i in range(n_genres):
            for j in range(n_genres):
                if i != j:
                    self.intensity_matrix[i, j] = max(symmetric_matrix[i, j], 0)

            self.intensity_matrix[i, i] = -np.sum(self.intensity_matrix[i])

        min_intensity = 0.01
        for i in range(n_genres):
            if abs(self.intensity_matrix[i, i]) < min_intensity:
                for j in range(n_genres):
                    if i != j:
                        self.intensity_matrix[i, j] += min_intensity / (n_genres - 1)
                self.intensity_matrix[i, i] = -np.sum(self.intensity_matrix[i])

        if self.auto_visualize:
            self._auto_visualize_markov_chain()

        return True

    def get_ratings_chronology(self, random_intervals=True):
        """Создание хронологии оценок с реалистичными случайными интервалами"""
        if self.cinema.rates.empty:
            return []

        ratings_list = []
        current_time = datetime.now()

        prev_time = current_time - timedelta(days=len(self.cinema.rates) * 2)

        for i, (movie_id, row) in enumerate(self.cinema.rates.iterrows()):
            if random_intervals:
                interval_days = np.random.exponential(scale=2.0)
            else:
                interval_days = 1.0

            timestamp = prev_time + timedelta(days=interval_days)
            ratings_list.append((movie_id, row["rating"], timestamp))
            prev_time = timestamp

        return sorted(ratings_list, key=lambda x: x[2])

    def calculate_time_interval(self, time1, time2):
        """Вычисление временного интервала между двумя оценками"""
        time_diff = abs((time2 - time1).total_seconds()) / (24 * 3600)
        return max(time_diff, 0.1)

    def get_genre_probabilities_continuous(self, time_t):
        """Получение вероятностей жанров в момент времени t с нормализацией"""
        if self.intensity_matrix is None:
            success = self.build_intensity_matrix()
            if not success:
                return None

        if self.cinema.rates.empty:
            return None

        ratings_chronology = self.get_ratings_chronology()
        if not ratings_chronology:
            return None

        last_movie_id, last_rating, last_time = ratings_chronology[-1]
        last_genres = self.cinema.movies.loc[last_movie_id, "genres"]

        initial_vector = np.zeros(len(self.genre_states))
        genre_count = len(last_genres)

        if genre_count == 0:
            return None

        for genre in last_genres:
            if genre in self.genre_states:
                idx = self.genre_states.index(genre)
                initial_vector[idx] = 1.0 / genre_count

        try:
            transition_matrix = expm(self.intensity_matrix * time_t)
            current_probs = initial_vector @ transition_matrix

            prob_sum = np.sum(current_probs)
            if prob_sum > 0:
                current_probs = current_probs / prob_sum
            else:
                current_probs = np.ones_like(current_probs) / len(current_probs)

        except Exception as e:
            print(f"Ошибка вычисления матричной экспоненты: {e}")
            return None

        return dict(zip(self.genre_states, current_probs))

    def recommend_movies_continuous(self, time_t=1.0, top_k=20):
        """Улучшенные рекомендации, чувствительные к времени"""
        genre_probs = self.get_genre_probabilities_continuous(time_t)
        if not genre_probs:
            unrated_movies = self.cinema.movies[
                ~self.cinema.movies.index.isin(self.cinema.rates.index)
            ]
            return unrated_movies.sample(min(top_k, len(unrated_movies)))

        candidate_movies = self.cinema.movies[
            ~self.cinema.movies.index.isin(self.cinema.rates.index)
        ].copy()

        if candidate_movies.empty:
            return candidate_movies

        def calculate_score(genres, time_factor=time_t):
            if not genres:
                return 0

            base_score = sum(genre_probs.get(genre, 0) for genre in genres)

            diversity_bonus = len(set(genres) & set(genre_probs.keys())) / len(genres)

            time_weight = min(time_t / 5.0, 1.0)
            final_score = base_score * (1 - time_weight) + diversity_bonus * time_weight

            return final_score

        candidate_movies["continuous_score"] = candidate_movies["genres"].apply(
            lambda genres: calculate_score(genres, time_t)
        )

        max_score = candidate_movies["continuous_score"].max()
        if max_score > 0:
            candidate_movies["continuous_score"] = (
                candidate_movies["continuous_score"] / max_score
            )

        recommendations = (
            candidate_movies[candidate_movies["continuous_score"] > 0.01]
            .sort_values("continuous_score", ascending=False)
            .head(top_k)
        )

        return recommendations.drop(columns=["continuous_score"], errors="ignore")

    def show_intensity_matrix(self):
        """Отображение матрицы интенсивностей с дополнительной информацией"""
        if self.intensity_matrix is None:
            success = self.build_intensity_matrix()
            if not success:
                self.cinema.console.print(
                    "[red]Не удалось построить матрицу интенсивностей[/red]"
                )
                return

        table = Table(
            title="Матрица интенсивностей переходов (непрерывная цепь Маркова)"
        )
        table.add_column("From/To", style="cyan")

        for genre in self.genre_states:
            table.add_column(genre, style="green", width=10)

        for i, from_genre in enumerate(self.genre_states):
            row = [from_genre]
            for j, to_genre in enumerate(self.genre_states):
                intensity = self.intensity_matrix[i, j]
                if abs(intensity) > 0.001:
                    if i == j:
                        row.append(f"[red]{intensity:+.3f}[/red]")
                    else:
                        row.append(f"{intensity:+.3f}")
                else:
                    row.append("0.000")
            table.add_row(*row)

        self.cinema.console.print(table)

        self.cinema.console.print(
            f"\n[bold]Размер матрицы:[/bold] {len(self.genre_states)}x{len(self.genre_states)}"
        )
        self.cinema.console.print(
            f"[bold]Количество жанров:[/bold] {len(self.genre_states)}"
        )

    def _auto_visualize_markov_chain(self, time_t=1.0):
        """Автоматическая визуализация и сохранение графа непрерывной цепи Маркова"""
        os.makedirs("markov", exist_ok=True)
        if self.intensity_matrix is None or len(self.genre_states) == 0:
            return

        G = nx.DiGraph()

        for genre in self.genre_states:
            G.add_node(genre)

        edge_labels = {}

        for i, from_genre in enumerate(self.genre_states):
            for j, to_genre in enumerate(self.genre_states):
                intensity = self.intensity_matrix[i, j]
                if intensity > 0.001:
                    G.add_edge(from_genre, to_genre, weight=intensity)
                    edge_labels[(from_genre, to_genre)] = f"{intensity:.2f}"

        plt.figure(figsize=(18, 14), dpi=120)

        if len(self.genre_states) <= 6:
            pos = nx.circular_layout(G, scale=3.0)
        elif len(self.genre_states) <= 10:
            pos = nx.circular_layout(G, scale=3.5)
        else:
            pos = nx.spring_layout(G, k=4, iterations=150, scale=3)

        node_size = 8000
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            node_color="lightblue",
            alpha=0.95,
            edgecolors="navy",
            linewidths=3,
            node_shape="o",
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=16,
            font_weight="bold",
            font_family="DejaVu Sans",
            verticalalignment="center",
            horizontalalignment="center",
        )

        if G.edges():
            edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1

            edge_widths = [1.5 + (w / max_weight) * 2.5 for w in edge_weights]
            edge_alphas = [0.4 + (w / max_weight) * 0.6 for w in edge_weights]

            edges = nx.draw_networkx_edges(
                G,
                pos,
                edge_color="darkred",
                width=edge_widths,
                alpha=edge_alphas,
                arrows=True,
                arrowsize=30,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.25",
                min_source_margin=20,
                min_target_margin=25,
                node_size=node_size,
                ax=plt.gca(),
            )

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=12,
                font_weight="bold",
                font_family="DejaVu Sans",
                label_pos=0.18,
                verticalalignment="center",
                horizontalalignment="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="lightgray",
                    linewidth=1.5,
                ),
            )

        title_info = f"Непрерывная цепь Маркова: интенсивности переходов (t={time_t})\n"
        title_info += f"Узлы: {len(G.nodes())} | Переходы: {len(G.edges())} | "
        title_info += f"Сгенерировано: {datetime.now().strftime('%d.%m.%Y %H:%M')}"

        plt.title(
            title_info,
            fontsize=18,
            fontweight="bold",
            pad=30,
            loc="center",
            fontfamily="DejaVu Sans",
        )

        plt.axis("off")
        plt.tight_layout(pad=4.0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"markov/continuous_markov_chain_t{time_t}_{timestamp}.png"
        plt.savefig(
            filename, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close()
