import pandas as pd


class KnowledgeBasedRecommender:
    def __init__(self, cinema):
        self.cinema = cinema

        self.current_candidates = None

        self.filters = {
            "min_year": None,
            "max_year": None,
            "min_popularity": None,
            "max_popularity": None,
            "min_rating": None,
            "max_rating": None,
            "genre_boost": {},
        }

    def start_recommendation(self):
        movies = self.cinema.movies.copy()

        ratings = self.cinema.ratings.reset_index()
        avg_rating = ratings.groupby("movieId")["rating"].mean()
        rating_count = ratings.groupby("movieId")["rating"].count()

        movies["rating"] = movies.index.map(avg_rating).fillna(0)
        movies["popularity"] = movies.index.map(rating_count).fillna(0)

        self.current_candidates = movies
        return self.get_top_unfiltered()

    def apply_critique(self, critique):
        critique = critique.lower().strip()

        if "нов" in critique or "new" in critique:
            self.filters["min_year"] = self.filters["min_year"] or 2000

        elif "стар" in critique or "old" in critique:
            self.filters["max_year"] = self.filters["max_year"] or 1995

        elif "популярн" in critique and "не" not in critique:
            self.filters["min_popularity"] = self.filters["min_popularity"] or 1000

        elif "непоп" in critique or "мало популяр" in critique:
            self.filters["max_popularity"] = self.filters["max_popularity"] or 200

        elif "высок" in critique and "рейтинг" in critique:
            self.filters["min_rating"] = self.filters["min_rating"] or 4.0

        elif "низк" in critique and "рейтинг" in critique:
            self.filters["max_rating"] = self.filters["max_rating"] or 3.0

        elif "больше" in critique:
            genre = critique.replace("больше", "").strip().capitalize()
            self.filters["genre_boost"][genre] = (
                self.filters["genre_boost"].get(genre, 0) + 1
            )

        elif "меньше" in critique:
            genre = critique.replace("меньше", "").strip().capitalize()
            self.filters["genre_boost"][genre] = (
                self.filters["genre_boost"].get(genre, 0) - 1
            )

        self.update_candidates()
        return self.get_top_filtered()

    def update_candidates(self):
        df = self.current_candidates.copy()

        f = self.filters

        if f["min_year"] is not None:
            df = df[df["year"] >= f["min_year"]]

        if f["max_year"] is not None:
            df = df[df["year"] <= f["max_year"]]

        if f["min_popularity"] is not None:
            df = df[df["popularity"] >= f["min_popularity"]]

        if f["max_popularity"] is not None:
            df = df[df["popularity"] <= f["max_popularity"]]

        if f["min_rating"] is not None:
            df = df[df["rating"] >= f["min_rating"]]

        if f["max_rating"] is not None:
            df = df[df["rating"] <= f["max_rating"]]

        if f["genre_boost"]:
            df["genre_score"] = 0
            for genre, weight in f["genre_boost"].items():
                df["genre_score"] += df["genres"].apply(
                    lambda g: weight if genre in g else 0
                )
            df = df.sort_values("genre_score", ascending=False)

        self.current_candidates = df

    def get_top_unfiltered(self, k=200000):
        """
        Первоначальная выдача рекомендаций
        """
        df = self.current_candidates.sort_values(
            ["rating", "popularity"], ascending=[False, False]
        )
        return df.head(k)

    def get_top_filtered(self, k=20):
        """
        Выдача рекомендаций после критики
        """
        if self.current_candidates.empty:
            return pd.DataFrame()
        return self.current_candidates.head(k)

    def genre_critique_map(self):
        genres = sorted(
            {g for movie_genres in self.cinema.movies["genres"] for g in movie_genres}
        )

        mapping = {}

        start_id = 7
        current = start_id

        for genre in genres:
            mapping[current] = (genre, +1)
            mapping[current + 1] = (genre, -1)
            current += 2

        return mapping

    def apply_critique_by_id(self, critique_id, top_k=20):
        if critique_id == 1:
            years = self.current_candidates["year"]
            if len(years) > 0:
                threshold = years.median()
                self.filters["min_year"] = (
                    threshold
                    if self.filters["min_year"] is None
                    else max(self.filters["min_year"], threshold)
                )

        elif critique_id == 2:
            years = self.current_candidates["year"]
            if len(years) > 0:
                threshold = years.median()
                self.filters["max_year"] = (
                    threshold
                    if self.filters["max_year"] is None
                    else min(self.filters["max_year"], threshold)
                )

        elif critique_id == 3:
            pops = self.current_candidates["popularity"]
            if len(pops) > 0:
                threshold = pops.median()
                self.filters["min_popularity"] = (
                    threshold
                    if self.filters["min_popularity"] is None
                    else max(self.filters["min_popularity"], threshold)
                )

        elif critique_id == 4:
            pops = self.current_candidates["popularity"]
            if len(pops) > 0:
                threshold = pops.median()
                self.filters["max_popularity"] = (
                    threshold
                    if self.filters["max_popularity"] is None
                    else min(self.filters["max_popularity"], threshold)
                )

        elif critique_id == 5:
            ratings = self.current_candidates["rating"]
            if len(ratings) > 0:
                threshold = ratings.median()
                self.filters["min_rating"] = (
                    threshold
                    if self.filters["min_rating"] is None
                    else max(self.filters["min_rating"], threshold)
                )

        elif critique_id == 6:
            ratings = self.current_candidates["rating"]
            if len(ratings) > 0:
                threshold = ratings.median()
                self.filters["max_rating"] = (
                    threshold
                    if self.filters["max_rating"] is None
                    else min(self.filters["max_rating"], threshold)
                )

        else:
            genre_actions = self.genre_critique_map()
            if critique_id in genre_actions:
                genre, weight = genre_actions[critique_id]
                self.filters["genre_boost"][genre] = (
                    self.filters["genre_boost"].get(genre, 0) + weight
                )

        self.update_candidates()
        return self.get_top_filtered(k=top_k)
