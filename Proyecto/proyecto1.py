import pandas as pd 
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import gradio as gr
import pickle
from dotenv import load_dotenv
import os
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()
# Configurar el motor de OpenAI
engine = "gpt-3.5-turbo"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#Descargamos los datos de https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset, donde tenemos tres ficheros distintos.
keywords = pd.read_csv('C:\\Users\\Admin\\Desktop\\Ejercicios\\Ejerciciosai\\Proyecto\\keywords.csv')
movies = pd.read_csv('C:\\Users\\Admin\\Desktop\\Ejercicios\\Ejerciciosai\\Proyecto\\movies_metadata.csv').\
                     drop(['belongs_to_collection', 'homepage', 'imdb_id', 'poster_path', 'status', 'video'], axis=1).\
                     drop([19730, 29503, 35587]) # Incorrect data type

movies['id'] = movies['id'].astype('int64')
#A continuación vamos a juntar ambos dataset
# Fusionar los DataFrames en función de la columna 'id'
keywords['id'] = keywords['id'].astype(int)
merged_df = pd.merge(keywords, movies, on='id', how='outer')  # Fusión del resultado con df3
print(merged_df)
#Nos encargamos de borrar los espacios en blanco, así como cambiar determinadas columnas.
merged_df.dropna(inplace=True)
merged_df['original_language'] = merged_df['original_language'].fillna('')
merged_df['runtime'] = merged_df['runtime'].fillna(0)
merged_df['tagline'] = merged_df['tagline'].fillna('')
percentil = merged_df['vote_count'].quantile(0.8)
print(percentil)
# Calculamos de igual forma la media de los votos y el número de votos
vote_average1 = merged_df['vote_average'].mean()
vote_count1 = merged_df['vote_count'].mean()
print(vote_average1, vote_count1)

print(merged_df.head())
merged_df.info()
columnas = merged_df.columns
print("Columnas del conjunto de datos:")
print(columnas)
# Histograma de 'vote_average'
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['vote_average'], bins=30, kde=True)
plt.title('Distribución de vote_average')
plt.show()

# Boxplot de 'vote_count'
plt.figure(figsize=(10, 6))
sns.boxplot(x=merged_df['vote_count'])
plt.title('Boxplot de vote_count')
plt.show()

# Scatter plot de 'vote_count' vs 'vote_average'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_count', y='vote_average', data=merged_df)
plt.title('Relación entre vote_count y vote_average')
plt.show()

# Gráfico de barras de distribución de 'original_language'
plt.figure(figsize=(12, 6))
sns.countplot(x='original_language', data=merged_df)
plt.title('Distribución de original_language')
plt.show()
#Vamos a ver la tabla de contenidos ratings_df
print("Añadiendo ratings para poder hacer el sistema de recomendación")
ratings_df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Ejercicios\\Ejerciciosai\\Proyecto\\ratings_small.csv')
ratings_df.drop('timestamp', axis=1, inplace=True)
ratings_df = ratings_df.merge(merged_df[['id', 'title', 'genres', 'overview', 'production_companies']], left_on='movieId',right_on='id', how='left')
ratings_df = ratings_df[~ratings_df['id'].isna()]
ratings_df.drop('id', axis=1, inplace=True)
ratings_df.reset_index(drop=True, inplace=True)
print(ratings_df.head())
# Scatter plot de 'vote_count' vs 'vote_average'
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_count', y='vote_average', data=merged_df)
plt.title('Relación entre vote_count y vote_average')
plt.show()

# Descargar recursos necesarios para NLTK
nltk.download('stopwords')

# Cargar el modelo de spaCy para español
nlp_es = spacy.load('es_core_news_sm')

# Tokenización, eliminación de stopwords y stemming con NLTK
def preprocess_text(text):
    # Tokenización con spaCy
    doc = nlp_es(text)
    tokens = [token.text for token in doc]

    # Eliminación de stopwords con NLTK
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Stemming con NLTK
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens

# Aplicar la función de preprocesamiento a una columna de texto en el DataFrame
# ratings_df['overview_processed'] = ratings_df['overview'].apply(preprocess_text)

# Mostrar resultados
# print("Texto original:")
# print(ratings_df['overview'].iloc[0])
# print("\nTexto procesado:")
# print(ratings_df['overview_processed'].iloc[0])
# Extract the "name" values from the list of dictionaries
# Combine the text from "cast," "genres," and "keywords" columns into a single text column
ratings_df['combined_text'] = ratings_df['overview'] + ' ' + ratings_df['genres']
def get_completion1(title, temperature=0):
    prompt = f"Dame una breve review de la película: {title}"
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

def get_completion(title, temperature=0):
    prompt = f"Dame el actor principal: {title}"
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

def train_recommendation():

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the combined text column
    tfidf_matrix = tfidf_vectorizer.fit_transform(ratings_df['combined_text'])

    save_recomendation(tfidf_matrix)
    return tfidf_matrix

def save_recomendation(tfidf_matrix):
    with open('tfidf_matrix.pickle', 'wb') as handle:
        pickle.dump(tfidf_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tfidf_matrix():

    with open('tfidf_matrix.pickle', 'rb') as handle:
        tfidf_matrix = pickle.load(handle)
    return tfidf_matrix


def recommendation(reference_movie_index):
    try:
        tfidf_matrix = load_tfidf_matrix()
    except:
        tfidf_matrix = train_recommendation()

    cosine_sim_scores = cosine_similarity(tfidf_matrix[reference_movie_index], tfidf_matrix)
    similar_movie_indices = cosine_sim_scores.argsort()[0][::-1][1:]

    top_N = 10
    reference_movie_title = ratings_df.iloc[reference_movie_index]['title']

    filtered_indices = [index for index in similar_movie_indices if ratings_df.iloc[index]['title'] != reference_movie_title]

    recommended_movies = []
    recommended_titles = set()  # Lista para rastrear títulos recomendados y evitar repeticiones

    for index in filtered_indices:
        movie_title = ratings_df.iloc[index]['title']
        if movie_title not in recommended_titles:
            movie_summary = get_completion(movie_title)
            recommended_movies.append((movie_title, movie_summary))
            recommended_titles.add(movie_title)

        if len(recommended_movies) == top_N:  # Salir del bucle cuando se alcanzan las 10 películas
            break

    return recommended_movies

# Interfaz Gradio
movie_options = [(row['title'], index) for index, row in ratings_df.iterrows()]

def gr_recommendation(movie_index):
    recommended_movies = recommendation(movie_index)
    output_text = ""
    for title, summary in recommended_movies:
        output_text += f"\n\nTítulo: {title}\nResumen: {summary}\n"
    return {'output': output_text}

# iface = gr.Interface( fn=gr_recommendation,inputs=gr.Dropdown(choices=movie_options, label="Película", info="Selecciona una película que te haya gustado"),outputs="text",)

#iface.launch(share=True)
#Vamos a modificar el código para que haga el sistema de recomendación con embeddings
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, save_model
# Combine el texto de las películas en un solo archivo de texto
corpus_file = 'corpus.txt'
ratings_df[['combined_text']].to_csv(corpus_file, index=False, header=False)

# Lee el archivo de texto
with open(corpus_file, 'r', encoding='utf-8') as f:
    corpus = [line.strip().split() for line in f]

# Divide el conjunto de datos en entrenamiento y prueba
train_corpus, test_corpus = train_test_split(corpus, test_size=0.2, random_state=42)

# Guarda el corpus de entrenamiento en un archivo
with open('corpus_train_processed.txt', 'w', encoding='utf-8') as f:
    for line in train_corpus:
        f.write(' '.join(line) + '\n')

# Entrena embeddings de Word2Vec
model = Word2Vec(sentences=train_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Guarda el modelo entrenado
model.save('movie_embeddings_w2v.model')

def load_movie_embeddings_w2v():
    model = Word2Vec.load('movie_embeddings_w2v.model')
    return model

def movie_embedding_w2v(movie_title):
    model = load_movie_embeddings_w2v()
    try:
        embedding = model.wv[movie_title]
        return embedding
    except KeyError:
        return np.zeros(model.vector_size)  # Devuelve un vector de ceros si la palabra no está en el vocabulario

def train_recommendation_w2v():
    embeddings = [movie_embedding_w2v(title) for title in ratings_df['title']]
    embeddings_matrix = np.vstack(embeddings)

    # Guardar el modelo entrenado
    model.save("recommendation_model.h5")
    return embeddings_matrix

def get_embeddings_from_model(model):
    # Suponiendo que la capa de embeddings es la primera capa del modelo
    embeddings_layer = model.layers[0]
    embeddings_matrix = embeddings_layer.get_weights()[0]
    return embeddings_matrix

def load_movie_embeddings_matrix_w2v():
    try:
        # Intentar cargar el modelo desde el archivo HDF5
        model = load_model("recommendation_model.h5")
        embeddings_matrix = get_embeddings_from_model(model)
    except (OSError, IOError):
        # Si falla la carga del modelo, entrenar y guardar nuevamente
        embeddings_matrix = train_recommendation_w2v()

    return embeddings_matrix


def recommendation_w2v(reference_movie_index):
    try:
        embeddings_matrix = load_movie_embeddings_matrix_w2v()
    except:
        embeddings_matrix = train_recommendation_w2v()

    cosine_sim_scores = cosine_similarity([embeddings_matrix[reference_movie_index]], embeddings_matrix)
    similar_movie_indices = cosine_sim_scores.argsort()[0][::-1][1:]

    top_N = 10
    reference_movie_title = ratings_df.iloc[reference_movie_index]['title']

    filtered_indices = [index for index in similar_movie_indices if ratings_df.iloc[index]['title'] != reference_movie_title]

    recommended_titles = set()
    recommended_movies = []

    for index in filtered_indices:
        movie_title = ratings_df.iloc[index]['title']
        if movie_title not in recommended_titles:
            recommended_movies.append(movie_title)
            recommended_titles.add(movie_title)

        if len(recommended_movies) == top_N:
            break

    return recommended_movies


def add_reviews_to_ratings(num_reviews=100):
    # Seleccionar las primeras 100 filas de ratings_df
    selected_ratings = ratings_df.head(num_reviews).copy()  # Copiar para evitar cambios en el DataFrame original

    # Añadir una nueva columna para las reviews generadas
    selected_ratings['generated_review'] = ""

    # Generar y añadir reviews para cada película
    for index, row in selected_ratings.iterrows():
        movie_title = row['title']
        generated_review = get_completion1(movie_title)
        print(f"Movie: {movie_title}, Generated Review: {generated_review}")
        selected_ratings.at[index, 'generated_review'] = generated_review

    # Añadir las nuevas columnas al DataFrame original
    ratings_df.loc[:num_reviews-1, 'generated_review'] = selected_ratings['generated_review']

# Llamar a la función para añadir reviews a las primeras 100 filas
add_reviews_to_ratings()

# Mostrar las primeras 100 filas de ratings_df con las nuevas columnas
print(ratings_df.head(100))

