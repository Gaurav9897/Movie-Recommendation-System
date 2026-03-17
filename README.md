# Movie Recommendation System (Genre-based, Skip-gram Word2Vec + Streamlit)

This project implements a **content-based movie recommendation system** using movie genres and a custom **skip-gram Word2Vec** model. Instead of relying on generic pre-trained embeddings (which often miss domain-specific words), the system learns task-specific genre embeddings directly from the dataset and uses cosine similarity to recommend semantically similar movies. An interactive Streamlit app allows users to select a movie and instantly see top-K recommendations.

---

## 1. Project Overview

- **Problem**: Recommend similar movies to a given title using only content (genres), without user ratings.
- **Dataset**: Movie metadata with `movieId`, `title`, and `genres` columns.
- **Approach**:
  - Tokenize movie genres.
  - Train a **skip-gram Word2Vec** model on these genre sequences.
  - Represent each movie as the average of its learned genre embeddings.
  - Use cosine similarity to recommend movies with the most similar vectors.
- **Outcome**: An end-to-end recommendation pipeline with a clean Python module and a Streamlit web app suitable for portfolio/demo use.

---

## 2. Dataset

The dataset contains one row per movie with:

- `movieId`: Unique movie identifier.
- `title`: Movie title (e.g., `"Toy Story (1995)"`).
- `genres`: Pipe-separated genres, e.g., `"Adventure|Animation|Children|Comedy|Fantasy"`.

For this project, the dataset is stored as:

```text
movies.csv

--

## 3. Methodology
3.1. Genre preprocessing
Split genres on the "|" character.

Lowercase and strip whitespace to produce genres_tokens such as:

"Comedy|Drama|Romance" → ["comedy", "drama", "romance"].

3.2. Skip-gram Word2Vec model
Build a corpus of genre token lists (one list per movie).

Train a skip-gram Word2Vec model (Gensim) on this corpus:

sg=1 → skip-gram architecture.

vector_size (embedding dimension) can be tuned (e.g., 50).

epochs can be increased for more training if needed.

This avoids missing-genre issues that occur with some pre-trained models and produces embeddings tailored to this movie dataset.

3.3. Movie representation
For each movie, fetch the embedding vector for each of its genres from the trained Word2Vec model.

Average these vectors to obtain a single dense vector per movie:

genre_embedding_avg ∈ ℝ^d.

3.4. Similarity and recommendation
Stack all movie vectors into an embedding matrix.

Given a query movie:

Locate its vector.

Compute cosine similarity with every other movie vector.

Sort by similarity and return the top-K movies (excluding the query itself).

-- 

## 4. Project Structure 

movie-recommendation-system/
│
├── Movie_Recommendation_System.ipynb   # Exploratory notebook (EDA, training, examples)
├── model_utils.py                      # Core module: data loading, training, embeddings, recommendations
├── app_streamlit.py                    # Streamlit web app UI
├── movies.csv                          # Movie metadata (movieId, title, genres)
├── README.md                           # This documentation
├── requirements.txt                    # Python dependencies

-- 

## 5. How to Run

5. How to Run
5.1. Environment setup

bash
# (Optional) Create and activate a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies

pip install -r requirements.txt
5.2. Run the notebook
bash
jupyter notebook
Open Movie_Recommendation_System.ipynb.

Run the cells to explore the dataset, see how the skip-gram model is trained, and view sample recommendations.

5.3. Run the Streamlit app

bash
streamlit run app_streamlit.py
Open the local URL shown in the terminal (usually http://localhost:8501).

Use the sidebar to select a movie and the number of recommendations.

-- 

## 6.  Example Usage

Inside the Streamlit app:

Pick a movie title from the dropdown (e.g., Toy Story (1995)).

Set the desired number of recommendations (e.g., 10).

Click “Get Recommendations”.

The app displays a table of similar movies with their genres and similarity scores.

--

## 7. Key Insights

Training a skip-gram Word2Vec model on genre tokens allows the system to learn dataset-specific relationships between genres instead of relying only on generic pre-trained vectors.

Simple content features (genres) combined with learned embeddings and cosine similarity can already yield meaningful recommendations.

This setup provides a solid foundation to incorporate richer text features like movie descriptions, tags, or plots.

-- 

## 8. Possible Extensions

Add movie descriptions, tags, or plot summaries and embed them using NLP methods (TF‑IDF, Word2Vec, BERT, etc.).

Combine this content-based approach with collaborative filtering (e.g., using user ratings) to build a hybrid recommendation system.

Deploy the Streamlit app publicly using Streamlit Community Cloud, Render, or Heroku.

Add filters in the UI (e.g., year range, minimum similarity threshold, specific genres).

--

## 9. Technologies Used

Python

pandas, numpy

Gensim (skip-gram Word2Vec)

scikit-learn (cosine similarity)

Streamlit (web app)

Jupyter Notebook

--

## 10. Author

Your Name: Gaurav Dhangar

Location: Gurugram, Haryana, India

LinkedIn: www.linkedin.com/in/gauravdhangar9

GitHub: https://github.com/Gaurav9897

