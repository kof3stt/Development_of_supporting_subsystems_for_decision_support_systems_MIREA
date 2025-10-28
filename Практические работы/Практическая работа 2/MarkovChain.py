import numpy as np
import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from rich.table import Table


class MarkovChainRecommender:
    """Внутренний класс для рекомендаций на основе цепи Маркова"""

    def __init__(self, cinema, auto_visualize=True):
        self.cinema = cinema
        self.transition_matrix = None
        self.genre_states = []
        self.auto_visualize = auto_visualize

        self._clean_markov_directory()

    def _clean_markov_directory(self):
        """Очистка папки markov при запуске программы"""
        try:
            if os.path.exists("markov"):
                shutil.rmtree("markov")
            os.makedirs("markov", exist_ok=True)
        except Exception as e:
            pass

    def reset_matrix(self):
        """Сброс матрицы переходов (вызывается при изменении оценок)"""
        self.transition_matrix = None
        self.genre_states = []

    def build_transition_matrix(self):
        """Построение матрицы переходов между жанрами с корректной нормализацией (вариант 1 — честная модель)"""
        if self.cinema.rates.empty:
            return False

        rated_movies = self.cinema.movies.loc[self.cinema.rates.index]

        all_genres = set()
        for genres in rated_movies["genres"]:
            all_genres.update(genres)
        self.genre_states = sorted(all_genres)

        if not self.genre_states:
            return False

        n_genres = len(self.genre_states)
        self.transition_matrix = np.zeros((n_genres, n_genres))
        state_to_idx = {genre: idx for idx, genre in enumerate(self.genre_states)}

        rated_indices = list(rated_movies.index)

        if len(rated_indices) < 2:
            return False

        for i in range(len(rated_indices) - 1):
            current_movie_id = rated_indices[i]
            next_movie_id = rated_indices[i + 1]

            current_genres = rated_movies.loc[current_movie_id, "genres"]
            next_genres = rated_movies.loc[next_movie_id, "genres"]

            if not current_genres or not next_genres:
                continue

            weight = 1.0 / len(next_genres)

            for curr_genre in current_genres:
                if curr_genre not in state_to_idx:
                    continue
                curr_idx = state_to_idx[curr_genre]

                for next_genre in next_genres:
                    if next_genre not in state_to_idx:
                        continue
                    next_idx = state_to_idx[next_genre]
                    self.transition_matrix[curr_idx, next_idx] += weight

        row_sums = self.transition_matrix.sum(axis=1)

        for i in range(n_genres):
            if row_sums[i] > 0:
                self.transition_matrix[i] /= row_sums[i]
            else:
                self.transition_matrix[i, i] = 1.0

        if self.auto_visualize:
            self._auto_visualize_markov_chain()
        return True

    def get_genre_probabilities(self, steps=1):
        """Получение вероятностей жанров через steps шагов"""
        if self.transition_matrix is None:
            success = self.build_transition_matrix()
            if not success:
                return None

        if self.cinema.rates.empty:
            return None

        rated_movies = self.cinema.movies.loc[self.cinema.rates.index]
        last_movie_id = rated_movies.index[-1]
        last_genres = rated_movies.loc[last_movie_id, "genres"]

        initial_vector = np.zeros(len(self.genre_states))
        genre_count = len(last_genres)

        if genre_count == 0:
            return None

        for genre in last_genres:
            if genre in self.genre_states:
                idx = self.genre_states.index(genre)
                initial_vector[idx] = 1.0 / genre_count

        current_probs = initial_vector
        for _ in range(steps):
            current_probs = current_probs @ self.transition_matrix

        return dict(zip(self.genre_states, current_probs))

    def recommend_movies(self, steps=1, top_k=20, min_probability=0.01):
        """Рекомендация фильмов на основе цепи Маркова"""
        genre_probs = self.get_genre_probabilities(steps)
        if not genre_probs:
            unrated_movies = self.cinema.movies[
                ~self.cinema.movies.index.isin(self.cinema.rates.index)
            ]
            return unrated_movies.sample(min(20, len(unrated_movies)))

        significant_genres = [
            genre for genre, prob in genre_probs.items() if prob >= min_probability
        ]

        if not significant_genres:
            significant_genres = [max(genre_probs, key=genre_probs.get)]

        candidate_movies = self.cinema.movies[
            ~self.cinema.movies.index.isin(self.cinema.rates.index)
        ].copy()

        if candidate_movies.empty:
            return candidate_movies

        def calculate_score(genres):
            return sum(genre_probs.get(genre, 0) for genre in genres)

        candidate_movies["markov_score"] = candidate_movies["genres"].apply(
            calculate_score
        )

        recommendations = (
            candidate_movies[candidate_movies["markov_score"] > 0]
            .sort_values("markov_score", ascending=False)
            .head(top_k)
        )

        return recommendations.drop(columns=["markov_score"], errors="ignore")

    def show_transition_matrix(self):
        """Отображение матрицы переходов"""
        if self.transition_matrix is None:
            success = self.build_transition_matrix()
            if not success:
                self.cinema.console.print(
                    "[red]Не удалось построить матрицу переходов[/red]"
                )
                return

        table = Table(title="Матрица переходов между жанрами")
        table.add_column("From/To", style="cyan")

        for genre in self.genre_states:
            table.add_column(genre, style="green", width=10)

        for i, from_genre in enumerate(self.genre_states):
            row = [from_genre]
            for j, to_genre in enumerate(self.genre_states):
                prob = self.transition_matrix[i, j]
                if prob > 0:
                    row.append(f"{prob:.3f}")
                else:
                    row.append("0")
            table.add_row(*row)

        self.cinema.console.print(table)

    def _auto_visualize_markov_chain(self, steps=1):
        """Автоматическое построение и сохранение графа цепи Маркова с финальными настройками"""
        os.makedirs("markov", exist_ok=True)
        if self.transition_matrix is None or len(self.genre_states) == 0:
            return

        G = nx.DiGraph()

        for genre in self.genre_states:
            G.add_node(genre)

        edge_labels = {}

        for i, from_genre in enumerate(self.genre_states):
            for j, to_genre in enumerate(self.genre_states):
                prob = self.transition_matrix[i, j]
                if prob >= 0.001:
                    G.add_edge(from_genre, to_genre, weight=prob)
                    edge_labels[(from_genre, to_genre)] = f"{prob:.2f}"

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

        title_info = f"Цепь Маркова: переходы между жанрами (k={steps})\n"
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
        filename = f"markov/markov_chain_k{steps}_{timestamp}.png"
        plt.savefig(
            filename, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close()
