import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


def load_movie_data(csv_path: str = "movies.csv") -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=["title", "genres"])
    return data


def preprocess_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genres_tokens"] = df["genres"].apply(
        lambda g: [token.strip().lower() for token in str(g).split("|") if token]
    )
    return df


def train_genre_word2vec(
    df: pd.DataFrame,
    vector_size: int = 50,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 100,
) -> Word2Vec:
    corpus = df["genres_tokens"].tolist()
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )
    return model


def get_genre_vector(word: str, model: Word2Vec) -> np.ndarray | None:
    if word in model.wv.key_to_index:
        return model.wv[word]
    return None


def compute_genre_embeddings(df: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    def average_embedding(tokens):
        vectors = [get_genre_vector(tok, model) for tok in tokens]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return np.zeros(model.vector_size, dtype=float)
        return np.mean(vectors, axis=0)

    df = df.copy()
    df["genre_embedding_avg"] = df["genres_tokens"].apply(average_embedding)
    return df


def build_embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    embeddings = np.stack(df["genre_embedding_avg"].values)
    return embeddings


def recommend_movies(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    movie_title: str,
    top_k: int = 10,
) -> pd.DataFrame:
    df = df.reset_index(drop=True)

    mask = df["title"].str.lower() == movie_title.lower()
    if not mask.any():
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    idx = df[mask].index[0]
    query_vec = embeddings[idx].reshape(1, -1)

    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    sim_scores[idx] = -1.0

    top_indices = np.argsort(sim_scores)[::-1][:top_k]

    results = df.loc[top_indices, ["movieId", "title", "genres"]].copy()
    results["similarity"] = sim_scores[top_indices]
    return results
