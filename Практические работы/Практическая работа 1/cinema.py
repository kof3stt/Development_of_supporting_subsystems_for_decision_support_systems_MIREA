import os
import re
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, Prompt
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt


def extract_year(title):
    title = title.strip()
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return int(match.group(1))
    return np.nan


class Genres(Enum):
    Action = "Action"
    Adventure = "Adventure"
    Animation = "Animation"
    Children = "Children's"
    Comedy = "Comedy"
    Crime = "Crime"
    Documentary = "Documentary"
    Drama = "Drama"
    Fantasy = "Fantasy"
    FilmNoir = "Film-Noir"
    Horror = "Horror"
    Musical = "Musical"
    Mystery = "Mystery"
    Romance = "Romance"
    SciFi = "Sci-Fi"
    Thriller = "Thriller"
    War = "War"
    Western = "Western"


class MovieLensCinema:
    def __init__(self, path, per_page=10):
        self.path = path
        self.links = None
        self.movies = None
        self.ratings = None
        self.tags = None
        self.console = Console()
        self.rates = pd.DataFrame(columns=["movieId", "rating"]).set_index("movieId")
        self.load_data()

        self.per_page = per_page

    def load_data(self):
        links_path = os.path.join(self.path, "links.csv")
        movies_path = os.path.join(self.path, "movies.csv")
        ratings_path = os.path.join(self.path, "ratings.csv")
        tags_path = os.path.join(self.path, "tags.csv")

        if os.path.exists(links_path):
            self.links = pd.read_csv(
                links_path,
                encoding="utf-8",
                index_col="movieId",
                dtype={"imdbId": "int64", "tmdbId": "Int64"},
            )
        else:
            raise FileNotFoundError("Не найдено файла links.csv")

        if os.path.exists(movies_path):
            self.movies = pd.read_csv(
                movies_path, encoding="utf-8", index_col="movieId", quotechar='"'
            )
            self.movies["title"] = self.movies["title"].str.strip()
            self.movies["genres"] = self.movies["genres"].apply(
                lambda x: [] if x == "(no genres listed)" else x.split("|")
            )
            self.movies["year"] = self.movies["title"].apply(extract_year)
            self.movies["year"] = self.movies["year"].astype("Int64")
            self.movies["title"] = self.movies["title"].apply(
                lambda t: re.sub(r"\s*\(\d{4}\)$", "", t)
            )
        else:
            raise FileNotFoundError("Не найдено файла movies.csv")

        if os.path.exists(ratings_path):
            self.ratings = pd.read_csv(
                ratings_path,
                encoding="utf-8",
                dtype={
                    "userId": "int32",
                    "movieId": "int32",
                    "rating": "float32",
                    "timestamp": "int64",
                },
            )
            self.ratings["datetime"] = pd.to_datetime(
                self.ratings["timestamp"], unit="s"
            )
            self.ratings = self.ratings.drop(columns=["timestamp"])
            self.ratings.set_index(["userId", "movieId"], inplace=True)

        else:
            raise FileNotFoundError("Не найдено файла ratings.csv")

        if os.path.exists(tags_path):
            self.tags = pd.read_csv(
                tags_path,
                dtype={
                    "userId": "int32",
                    "movieId": "int32",
                    "tag": "string",
                    "timestamp": "int64",
                },
                quotechar='"',
            )
            self.tags["datetime"] = pd.to_datetime(self.tags["timestamp"], unit="s")
            self.tags.set_index(["userId", "movieId"], inplace=True)
        else:
            raise FileNotFoundError("Не найдено файла tags.csv")

    def get_rating_matrix(self):
        df = self.ratings.reset_index()

        rating_matrix = df.pivot_table(
            index="userId", columns="movieId", values="rating"
        )

        return rating_matrix

    def show_movies(self, page=1):
        start = (page - 1) * self.per_page
        end = start + self.per_page
        subset = self.movies.iloc[start:end]

        table = Table(title=f"🎬 Список фильмов — страница {page}")

        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Название", style="bold")
        table.add_column("Год", justify="center", style="magenta")
        table.add_column("Жанры", style="green")

        for movie_id, row in subset.iterrows():
            genres_str = ", ".join(row["genres"]) if row["genres"] else "—"
            year_str = str(row["year"]) if not pd.isna(row["year"]) else "—"
            table.add_row(str(movie_id), row["title"], year_str, genres_str)

        self.console.print(table)

    def run(self):
        page = 1
        total_pages = (len(self.movies) - 1) // self.per_page + 1

        while True:
            os.system("cls" if os.name == "nt" else "clear")
            self.show_movies(page=page)

            self.show_paginator(page, total_pages)

            self.console.print(
                "[yellow]n[/yellow] — следующая, [yellow]p[/yellow] — предыдущая, "
                "[yellow]число[/yellow] — перейти на страницу\n"
                "[yellow]f[/yellow] — поиск\n"
                "[yellow]s[/yellow] — отсортировать результаты\n"
                "[yellow]o[/yellow] — перейти на страницу фильма\n"
                "[yellow]b[/yellow] — вернуться на главную страницу\n"
                "[yellow]q[/yellow] — выход"
            )

            choice = input("> ").strip().lower()
            if choice == "n" and page < total_pages:
                page += 1
            elif choice == "p" and page > 1:
                page -= 1
            elif choice.isdigit():
                num = int(choice)
                if 1 <= num <= total_pages:
                    page = num
            elif choice == "f":
                self.search_movies()
            elif choice == "s":
                self.sort_movies(self.movies)
            elif choice == "o":
                movie_id_str = Prompt.ask("Введите ID фильма (или 'b' для отмены)")
                if movie_id_str.lower() == "b":
                    continue
                if movie_id_str.isdigit():
                    self.show_movie_page(int(movie_id_str))
                else:
                    self.console.print("[red]Некорректный ID[/red]")
                    input("Нажмите Enter, чтобы вернуться...")
            elif choice == "b":
                break
            elif choice == "q":
                if Confirm.ask("Are you sure?"):
                    exit(0)

    def menu(self):
        while True:
            os.system("cls" if os.name == "nt" else "clear")

            self.console.print(
                Panel.fit(
                    "[bold magenta]🎬 Добро пожаловать в консольный кинотеатр![/bold magenta]\nВыберите действие:"
                ),
                justify="center",
            )

            self.console.print("[cyan]1.[/cyan] 📽 Подборка фильмов")
            self.console.print("[cyan]2.[/cyan] 🔍 Мои рекомендации")
            self.console.print("[cyan]3.[/cyan] 🔍 Мои оценки")
            self.console.print("[cyan]4.[/cyan] ❌ Выход")

            choice = Prompt.ask("\nВыберите пункт", choices=["1", "2", "3", "4"])

            if choice == "1":
                self.run()
            elif choice == "2":
                self.show_recommendations()
            elif choice == "3":
                self.show_my_ratings()
            elif choice == "4":
                if Confirm.ask("[bold red]Вы действительно хотите выйти?[/bold red]"):
                    break

    def show_movie_page(self, movie_id: int):
        """Страница фильма с информацией, тегами и возможностью оценки"""
        if movie_id not in self.movies.index:
            self.console.print(f"[red]Фильм с ID {movie_id} не найден.[/red]")
            input("Нажмите Enter, чтобы вернуться...")
            return

        while True:
            os.system("cls" if os.name == "nt" else "clear")

            movie = self.movies.loc[movie_id]

            info_table = Table(show_header=False, box=None)
            info_table.add_row(
                "🎬 Название:", f"[bold cyan]{movie['title']}[/bold cyan]"
            )
            info_table.add_row(
                "📅 Год:", str(movie["year"]) if movie["year"] == movie["year"] else "—"
            )
            info_table.add_row("🏷 Жанры:", f"[green]{movie['genres']}[/green]")
            self.console.print(Panel(info_table, title=f"ID {movie_id}", expand=False))

            if self.tags is not None:
                try:
                    df = self.tags.xs(movie_id, level="movieId")
                    movie_tags = df["tag"].tolist()
                except KeyError:
                    movie_tags = []

                if movie_tags:
                    tags_str = ", ".join(f"[yellow]{t}[/yellow]" for t in movie_tags)
                    self.console.print(Panel(tags_str, title="Теги", expand=False))
                else:
                    self.console.print(
                        Panel(
                            "[italic grey]Для этого фильма нет тегов[/italic grey]",
                            title="Теги",
                            expand=False,
                        )
                    )

            self.console.print(
                "\n[bold]Доступные действия:[/bold]\n"
                "[green]r[/green] — оценить фильм\n"
                "[yellow]b[/yellow] — назад\n"
                "[red]q[/red] — выйти из приложения"
            )

            choice = input("> ").strip().lower()
            if choice == "r":
                self.rate_movie(movie_id)
            elif choice == "b":
                break
            elif choice == "q":
                if Confirm.ask("Are you sure?"):
                    exit(0)

    def rate_movie(self, movie_id: int):
        """Простейшая система выставления оценки"""
        while True:
            rating = Prompt.ask(
                "Введите оценку от 0.5 до 5.0 (или 'b' чтобы вернуться)"
            )
            if rating.lower() == "b":
                break
            try:
                rating = float(rating)
                if 0.5 <= rating <= 5.0:
                    self.rates.loc[movie_id, "rating"] = rating
                    self.console.print(
                        f"[bold green]Спасибо! Ваша оценка {rating} сохранена для фильма c ID {movie_id}[/bold green]"
                    )
                    input("Нажмите Enter, чтобы вернуться...")
                    break
                else:
                    self.console.print(
                        "[red]Оценка должна быть в диапазоне 0.5–5.0[/red]"
                    )
                    input("Нажмите Enter, чтобы вернуться...")
                    break
            except ValueError:
                self.console.print("[red]Введите корректное число или 'b'[/red]")
                input("Нажмите Enter, чтобы вернуться...")
                break

    def search_movies(self):
        """Поиск фильмов по названию с фильтрацией по году и жанрам"""
        os.system("cls" if os.name == "nt" else "clear")
        search_query = (
            Prompt.ask("Введите название фильма (Enter — пропустить)").strip().lower()
        )
        selected_genres = self.choose_genres()
        year_op, year_val = self.choose_year_filter()

        def apply_filters():
            df = self.movies
            if search_query:
                df = df[df["title"].str.lower().str.contains(search_query, na=False)]
            if selected_genres:
                df = df[
                    df["genres"].apply(
                        lambda g: all(gen in g for gen in selected_genres)
                    )
                ]
            if year_op is not None and year_val is not None:
                if year_op == "=":
                    df = df[df["year"] == year_val]
                elif year_op == ">":
                    df = df[df["year"] > year_val]
                elif year_op == "<":
                    df = df[df["year"] < year_val]
            return df

        filtered_df = apply_filters()
        if filtered_df.empty:
            self.console.print(
                Panel("[red]❌ Фильмы не найдены[/red]", title="Результат")
            )
            input("Нажмите Enter, чтобы вернуться...")
            return

        page = 1
        per_page = self.per_page

        while True:
            os.system("cls" if os.name == "nt" else "clear")

            total_pages = max(1, (len(filtered_df) - 1) // per_page + 1)
            page = min(page, total_pages)

            start = (page - 1) * per_page
            end = start + per_page
            page_data = filtered_df.iloc[start:end]

            table = Table(
                title=f"Результаты поиска: {len(filtered_df)}",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("ID", style="cyan", width=6)
            table.add_column("Название", style="white")
            table.add_column("Год", style="yellow", width=8)
            table.add_column("Жанры", style="green")

            for idx, row in page_data.iterrows():
                year = str(row["year"]) if not pd.isna(row["year"]) else "—"
                genres = ", ".join(row["genres"]) if row["genres"] else "—"
                table.add_row(str(idx), row["title"], year, genres)

            self.console.print(table)

            self.show_paginator(page, total_pages)

            status = f"[bold]Фильтры:[/bold] Название: [cyan]{search_query or '—'}[/cyan], Жанры: [cyan]{', '.join(selected_genres) if selected_genres else '—'}[/cyan], Год: [cyan]{year_op + str(year_val) if year_op and year_val else '—'}[/cyan]"
            self.console.print(status)

            self.console.print(
                "[yellow]n[/yellow] — следующая, [yellow]p[/yellow] — предыдущая, [yellow]число[/yellow] — перейти на страницу\n"
                "[yellow]s[/yellow] — отсортировать результаты\n"
                "[yellow]o[/yellow] — открыть страницу фильма по ID\n"
                "[yellow]r[/yellow] — сброс фильтров\n"
                "[yellow]b[/yellow] — назад"
            )

            choice = input("> ").strip().lower()
            if choice == "n" and page < total_pages:
                page += 1
            elif choice == "p" and page > 1:
                page -= 1
            elif choice == "s":
                self.sort_movies(filtered_df)
            elif choice.isdigit():
                num = int(choice)
                if 1 <= num <= total_pages:
                    page = num
            elif choice == "o":
                movie_id_str = Prompt.ask("Введите ID фильма (или 'b' чтобы вернуться)")
                if movie_id_str.lower() == "b":
                    continue
                try:
                    movie_id = int(movie_id_str)
                    if movie_id in self.movies.index:
                        self.show_movie_page(movie_id)
                    else:
                        self.console.print("[red]Фильм с таким ID не найден[/red]")
                        input("Нажмите Enter, чтобы продолжить...")
                except ValueError:
                    self.console.print("[red]Введите корректный ID[/red]")
                    input("Нажмите Enter, чтобы продолжить...")
            elif choice == "r":
                search_query = ""
                selected_genres = []
                year_op, year_val = None, None
                filtered_df = self.movies
                page = 1
            elif choice == "b":
                break

    def choose_genres(self):
        """Выбор одного или нескольких жанров через консоль с таблицей"""
        genre_list = list(Genres)
        table = Table(
            title="Выберите жанры", show_header=True, header_style="bold magenta"
        )
        table.add_column("№", justify="center", style="cyan", width=4)
        table.add_column("Жанр", justify="left", style="green")

        for i, genre in enumerate(genre_list, 1):
            table.add_row(str(i), genre.value)

        self.console.print(table)
        choice = Prompt.ask(
            "Введите номера жанров через пробел (Enter — пропустить)"
        ).strip()
        if not choice:
            return []

        selected_genres = []
        for num in choice.split():
            num = num.strip()
            if num.isdigit():
                idx = int(num) - 1
                if 0 <= idx < len(genre_list):
                    selected_genres.append(genre_list[idx].value)
        return selected_genres

    def choose_year_filter(self):
        """Выбор фильтра по году с оператором"""
        op = Prompt.ask(
            "Выберите оператор для фильтра по году",
            choices=[">", "<", "="],
            default="=",
        )
        year_str = Prompt.ask("Введите год").strip()
        if not year_str.isdigit():
            self.console.print("[red]Некорректный год, фильтр не будет применён[/red]")
            return None, None
        return op, int(year_str)

    def show_paginator(self, page, total_pages):
        """Красивый центрированный пагинатор"""
        paginator_text = Text()
        last_was_ellipsis = False

        if page > 1:
            paginator_text.append("◀ Prev ", style="bold yellow")
        else:
            paginator_text.append("         ")

        for p in range(1, total_pages + 1):
            if p == 1 or p == total_pages or abs(p - page) <= 1:
                if p == page:
                    paginator_text.append(f" {p} ", style="reverse green")
                else:
                    paginator_text.append(f" {p} ", style="bold cyan")
                last_was_ellipsis = False
            else:
                if not last_was_ellipsis:
                    paginator_text.append(" ... ", style="dim")
                    last_was_ellipsis = True

        if page < total_pages:
            paginator_text.append(" Next ▶", style="bold yellow")

        self.console.print(Align.center(paginator_text))

    def show_my_ratings(self):
        if self.rates.empty:
            self.console.print(
                Panel(
                    "[red]Вы еще не оценили ни одного фильма[/red]", title="Ваши оценки"
                )
            )
            input("Нажмите Enter, чтобы вернуться...")
            return

        page = 1
        df = self.rates.merge(self.movies, on="movieId")
        total_pages = (len(df) - 1) // self.per_page + 1

        while len(df):
            os.system("cls" if os.name == "nt" else "clear")

            start = (page - 1) * self.per_page
            end = start + self.per_page
            subset = df.iloc[start:end]

            rates = Table(show_header=True, title=f"Ваши оценки — страница {page}")
            rates.add_column("ID", style="cyan", no_wrap=True)
            rates.add_column("Название", style="bold")
            rates.add_column("Год", justify="center", style="magenta")
            rates.add_column("Жанры", style="green")
            rates.add_column("Оценка", style="green")

            for movie_id, row in subset.iterrows():
                genres_str = ", ".join(row["genres"]) if row["genres"] else "—"
                year_str = str(row["year"]) if not pd.isna(row["year"]) else "—"
                rates.add_row(
                    str(movie_id),
                    row["title"],
                    year_str,
                    genres_str,
                    str(row["rating"]),
                )

            self.console.print(rates)

            self.show_paginator(page, total_pages)

            self.console.print(
                "[yellow]n[/yellow] — следующая, [yellow]p[/yellow] — предыдущая, "
                "[yellow]число[/yellow] — перейти на страницу\n"
                "[yellow]d[/yellow] — удалить оценку\n"
                "[yellow]o[/yellow] — перейти на страницу фильма\n"
                "[yellow]b[/yellow] — вернуться на главную страницу\n"
                "[yellow]q[/yellow] — выход"
            )

            choice = input("> ").strip().lower()
            if choice == "n" and page < total_pages:
                page += 1
            elif choice == "p" and page > 1:
                page -= 1
            elif choice.isdigit():
                num = int(choice)
                if 1 <= num <= total_pages:
                    page = num
            elif choice == "d":
                movie_id_str = Prompt.ask(
                    "Введите ID фильма, у которого хотите удалить оценку (или 'b' для отмены)"
                )
                if movie_id_str.lower() == "b":
                    continue
                if movie_id_str.isdigit():
                    id = int(movie_id_str)
                    if id in self.rates.index:
                        self.rates = self.rates.drop(index=id)
                        df = df.drop(index=id)
                    else:
                        self.console.print(
                            "[red]Не найдено оценки для фильма с заданным ID[/red]"
                        )
                        input("Нажмите Enter, чтобы вернуться...")
                else:
                    self.console.print("[red]Некорректный ID[/red]")
                    input("Нажмите Enter, чтобы вернуться...")
            elif choice == "o":
                movie_id_str = Prompt.ask("Введите ID фильма (или 'b' для отмены)")
                if movie_id_str.lower() == "b":
                    continue
                if movie_id_str.isdigit():
                    self.show_movie_page(int(movie_id_str))
                else:
                    self.console.print("[red]Некорректный ID[/red]")
                    input("Нажмите Enter, чтобы вернуться...")
            elif choice == "b":
                break
            elif choice == "q":
                if Confirm.ask("Are you sure?"):
                    exit(0)

    def sort_movies(self, frame):
        """Сортировка фильмов по различным полям"""
        sort_fields = [
            ("ID", "movieId"),
            ("Название", "title"),
            ("Год", "year"),
            ("Жанры", "genres"),
        ]

        table = Table(
            title="Выберите поле для сортировки",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("№", justify="center", style="cyan", width=4)
        table.add_column("Поле", style="green")

        for i, (label, _) in enumerate(sort_fields, 1):
            table.add_row(str(i), label)

        self.console.print(table)

        choice = Prompt.ask("Введите номер поля (или Enter для отмены)").strip()

        if not choice.isdigit():
            return

        choice_num = int(choice)
        if not 1 <= choice_num <= len(sort_fields):
            self.console.print("[red]Некорректный выбор[/red]")
            input("Нажмите Enter, чтобы вернуться...")
            return

        field_label, field_name = sort_fields[choice_num - 1]

        direction = Prompt.ask(
            "Выберите направление сортировки", choices=["asc", "desc"], default="asc"
        )

        ascending = direction == "asc"

        if field_name == "movieId":
            sorted_df = frame.sort_index(ascending=ascending)
        elif field_name == "genres":
            sorted_df = frame.copy()
            sorted_df["__sort_genre"] = sorted_df["genres"].apply(
                lambda g: g[0] if g else ""
            )
            sorted_df = sorted_df.sort_values("__sort_genre", ascending=ascending).drop(
                columns="__sort_genre"
            )
        else:
            sorted_df = frame.sort_values(field_name, ascending=ascending)

        self.paginated_view(
            sorted_df, title=f"Сортировка по {field_label} ({direction})"
        )

    def paginated_view(self, df, title="Список фильмов"):
        """Универсальный постраничный просмотр DataFrame фильмов"""
        page = 1
        total_pages = max(1, (len(df) - 1) // self.per_page + 1)

        while True:
            os.system("cls" if os.name == "nt" else "clear")
            start = (page - 1) * self.per_page
            end = start + self.per_page
            subset = df.iloc[start:end]

            table = Table(title=f"{title} — страница {page}", show_header=True)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Название", style="bold")
            table.add_column("Год", justify="center", style="magenta")
            table.add_column("Жанры", style="green")

            for movie_id, row in subset.iterrows():
                genres_str = ", ".join(row["genres"]) if row["genres"] else "—"
                year_str = str(row["year"]) if not pd.isna(row["year"]) else "—"
                table.add_row(str(movie_id), row["title"], year_str, genres_str)

            self.console.print(table)
            self.show_paginator(page, total_pages)
            self.console.print(
                "[yellow]n[/yellow] — следующая, [yellow]p[/yellow] — предыдущая, "
                "[yellow]число[/yellow] — перейти на страницу\n"
                "[yellow]o[/yellow] — открыть страницу фильма, [yellow]b[/yellow] — назад"
            )

            choice = input("> ").strip().lower()
            if choice == "n" and page < total_pages:
                page += 1
            elif choice == "p" and page > 1:
                page -= 1
            elif choice.isdigit():
                num = int(choice)
                if 1 <= num <= total_pages:
                    page = num
            elif choice == "o":
                movie_id_str = Prompt.ask("Введите ID фильма (или 'b' для отмены)")
                if movie_id_str.lower() == "b":
                    continue
                if movie_id_str.isdigit():
                    self.show_movie_page(int(movie_id_str))
            elif choice == "b":
                break
        os.system("cls" if os.name == "nt" else "clear")

    def show_recommendations(self):
        """Вывод рекомендаций для текущего пользователя, включая оценки из self.rates"""
        os.system("cls" if os.name == "nt" else "clear")

        R = self.get_rating_matrix()
        user_ids = R.index.tolist()
        movie_ids = R.columns.tolist()

        current_user_id = -1

        current_user_ratings = pd.Series(
            [np.nan] * len(movie_ids), index=movie_ids, dtype=float
        )

        for movie_id in self.rates.index:
            if movie_id in movie_ids:
                current_user_ratings[movie_id] = self.rates.loc[movie_id, "rating"]

        current_user_df = pd.DataFrame([current_user_ratings], index=[current_user_id])

        if current_user_df.notna().any().any():
            R = pd.concat([R, current_user_df])
            user_ids.append(current_user_id)
            current_user_idx = R.index.get_loc(current_user_id)
        else:
            self.console.print(
                Panel(
                    "[red]У вас нет оценок для рекомендаций. Сначала оцените несколько фильмов.[/red]",
                    title="Рекомендации",
                )
            )
            input("Нажмите Enter, чтобы вернуться...")
            return

        strategy = Prompt.ask(
            "Выберите стратегию [user-based/item-based]",
            choices=["user-based", "item-based"],
            default="user-based",
        )

        metric = Prompt.ask(
            "Выберите меру сходства [jaccard/lp/otiai/pearson]",
            choices=["jaccard", "lp", "otiai", "pearson"],
            default="pearson",
        )

        R_values = R.values.astype(np.float64)
        n_users, n_items = R_values.shape
        user_means = np.nanmean(R_values, axis=1)
        item_means = np.nanmean(R_values, axis=0)
        preds = np.full(n_items, np.nan)

        if strategy == "user-based":
            sims = np.zeros(n_users)
            current_ratings = R_values[current_user_idx, :]
            for u in range(n_users):
                if u == current_user_idx:
                    continue
                other_ratings = R_values[u, :]
                mask = ~np.isnan(current_ratings) & ~np.isnan(other_ratings)
                if np.sum(mask) == 0:
                    sims[u] = 0
                    continue
                if metric == "pearson":
                    sims[u] = np.corrcoef(current_ratings[mask], other_ratings[mask])[
                        0, 1
                    ]
                elif metric == "lp":
                    sims[u] = -np.linalg.norm(
                        current_ratings[mask] - other_ratings[mask], ord=2
                    )
                elif metric == "jaccard":
                    sims[u] = np.sum(
                        (current_ratings[mask] > 0) & (other_ratings[mask] > 0)
                    ) / np.sum((current_ratings[mask] > 0) | (other_ratings[mask] > 0))
                elif metric == "otiai":
                    sims[u] = np.sum(current_ratings[mask] * other_ratings[mask]) / (
                        np.linalg.norm(current_ratings[mask])
                        * np.linalg.norm(other_ratings[mask])
                    )
                else:
                    sims[u] = 0

            for i in range(n_items):
                if not np.isnan(R_values[current_user_idx, i]):
                    continue
                mask = ~np.isnan(R_values[:, i])
                if np.sum(mask) == 0:
                    continue
                numerator = np.sum(sims[mask] * (R_values[mask, i] - user_means[mask]))
                denominator = np.sum(np.abs(sims[mask])) + 1e-8
                preds[i] = user_means[current_user_idx] + numerator / denominator

        else:
            for i in range(n_items):
                if not np.isnan(R_values[current_user_idx, i]):
                    continue
                rated_mask = ~np.isnan(R_values[current_user_idx, :])
                sims_i = []
                ratings_i = []
                for j in np.where(rated_mask)[0]:
                    mask = ~np.isnan(R_values[:, i]) & ~np.isnan(R_values[:, j])
                    if np.sum(mask) == 0:
                        sim = 0
                    else:
                        if metric == "pearson":
                            sim = np.corrcoef(R_values[mask, i], R_values[mask, j])[
                                0, 1
                            ]
                        elif metric == "lp":
                            sim = -np.linalg.norm(
                                R_values[mask, i] - R_values[mask, j], ord=2
                            )
                        elif metric == "jaccard":
                            sim = np.sum(
                                (R_values[mask, i] > 0) & (R_values[mask, j] > 0)
                            ) / np.sum(
                                (R_values[mask, i] > 0) | (R_values[mask, j] > 0)
                            )
                        elif metric == "otiai":
                            sim = np.sum(R_values[mask, i] * R_values[mask, j]) / (
                                np.linalg.norm(R_values[mask, i])
                                * np.linalg.norm(R_values[mask, j])
                            )
                        else:
                            sim = 0
                    sims_i.append(sim)
                    ratings_i.append(R_values[current_user_idx, j] - item_means[j])
                sims_i = np.array(sims_i)
                ratings_i = np.array(ratings_i)
                if np.sum(np.abs(sims_i)) > 0:
                    preds[i] = item_means[i] + np.dot(sims_i, ratings_i) / np.sum(
                        np.abs(sims_i)
                    )

        recs_df = pd.DataFrame({"movieId": movie_ids, "pred_rating": preds})
        recs_df = recs_df.dropna().sort_values("pred_rating", ascending=False)

        self.paginated_view(
            df=self.movies.loc[recs_df["movieId"]],
            title=f"Рекомендации ({strategy}, {metric})",
        )

    def plot_similarity_metrics(
        self, strategy: str = "user-based", test_ratio: float = 0.2
    ):
        metrics = ["pearson", "lp", "jaccard", "otiai"]
        results = []

        for metric in metrics:
            rmse, mae = self._compute_metrics_for_plot(metric, strategy, test_ratio)
            results.append({"metric": metric, "RMSE": rmse, "MAE": mae})

        df = pd.DataFrame(results)

        sns.set_theme(style="whitegrid", font_scale=1.1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.barplot(x="metric", y="RMSE", data=df, ax=axes[0], palette="Blues_d")
        axes[0].set_title(f"Сравнение RMSE ({strategy})")
        axes[0].set_xlabel("Метрика близости")
        axes[0].set_ylabel("RMSE")

        sns.barplot(x="metric", y="MAE", data=df, ax=axes[1], palette="Greens_d")
        axes[1].set_title(f"Сравнение MAE ({strategy})")
        axes[1].set_xlabel("Метрика близости")
        axes[1].set_ylabel("MAE")

        plt.tight_layout()
        plt.show()

    def _compute_metrics_for_plot(self, metric: str, strategy: str, test_ratio: float):
        R = self.get_rating_matrix().copy().values.astype(float)
        n_users, n_items = R.shape

        rng = np.random.default_rng(42)
        train = R.copy()
        test_mask = ~np.isnan(R) & (rng.random(R.shape) < test_ratio)
        test_true = np.full_like(R, np.nan)
        test_true[test_mask] = R[test_mask]
        train[test_mask] = np.nan

        user_means = np.nanmean(train, axis=1)
        item_means = np.nanmean(train, axis=0)
        preds = np.full_like(R, np.nan)

        for u in range(n_users):
            if strategy == "user-based":
                sims = np.zeros(n_users)
                current = train[u, :]
                for v in range(n_users):
                    if v == u:
                        continue
                    other = train[v, :]
                    mask = ~np.isnan(current) & ~np.isnan(other)
                    if np.sum(mask) == 0:
                        sims[v] = 0
                        continue
                    if metric == "pearson":
                        sims[v] = np.corrcoef(current[mask], other[mask])[0, 1]
                    elif metric == "lp":
                        sims[v] = -np.linalg.norm(current[mask] - other[mask])
                    elif metric == "jaccard":
                        sims[v] = np.sum(
                            (current[mask] > 0) & (other[mask] > 0)
                        ) / np.sum((current[mask] > 0) | (other[mask] > 0))
                    elif metric == "otiai":
                        sims[v] = np.sum(current[mask] * other[mask]) / (
                            np.linalg.norm(current[mask]) * np.linalg.norm(other[mask])
                        )
                    else:
                        sims[v] = 0

                for i in range(n_items):
                    if not np.isnan(train[u, i]):
                        continue
                    mask = ~np.isnan(train[:, i])
                    if np.sum(mask) == 0:
                        continue
                    numerator = np.sum(sims[mask] * (train[mask, i] - user_means[mask]))
                    denominator = np.sum(np.abs(sims[mask])) + 1e-8
                    preds[u, i] = user_means[u] + numerator / denominator

            else:
                for i in range(n_items):
                    if not np.isnan(train[u, i]):
                        continue
                    rated_mask = ~np.isnan(train[u, :])
                    sims_i = []
                    ratings_i = []
                    for j in np.where(rated_mask)[0]:
                        mask = ~np.isnan(train[:, i]) & ~np.isnan(train[:, j])
                        if np.sum(mask) == 0:
                            sim = 0
                        else:
                            if metric == "pearson":
                                sim = np.corrcoef(train[mask, i], train[mask, j])[0, 1]
                            elif metric == "lp":
                                sim = -np.linalg.norm(train[mask, i] - train[mask, j])
                            elif metric == "jaccard":
                                sim = np.sum(
                                    (train[mask, i] > 0) & (train[mask, j] > 0)
                                ) / np.sum((train[mask, i] > 0) | (train[mask, j] > 0))
                            elif metric == "otiai":
                                sim = np.sum(train[mask, i] * train[mask, j]) / (
                                    np.linalg.norm(train[mask, i])
                                    * np.linalg.norm(train[mask, j])
                                )
                            else:
                                sim = 0
                        sims_i.append(sim)
                        ratings_i.append(train[u, j] - item_means[j])
                    sims_i = np.array(sims_i)
                    ratings_i = np.array(ratings_i)
                    if np.sum(np.abs(sims_i)) > 0:
                        preds[u, i] = item_means[i] + np.dot(
                            sims_i, ratings_i
                        ) / np.sum(np.abs(sims_i))

        mask_eval = ~np.isnan(test_true) & ~np.isnan(preds)
        if np.sum(mask_eval) == 0:
            return np.nan, np.nan

        diff = preds[mask_eval] - test_true[mask_eval]
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        return rmse, mae


if __name__ == "__main__":
    cinema = MovieLensCinema(r"ml-latest-small")
    cinema.menu()
    # cinema.plot_similarity_metrics()
