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
# Quitar los comentarios para ejecutar la interfaz de radio.
# iface = gr.Interface( fn=gr_recommendation,inputs=gr.Dropdown(choices=movie_options, label="Película", info="Selecciona una película que te haya gustado"),outputs="text",)

#iface.launch(share=True)

#Vamos a modificar el código para que haga el sistema de recomendación con embeddings.
#No funciona el radio ya que tarda más de 1 min en cargar.
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

#Funcion para añadir reviews usando la api de OpenAi
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

# Llamar a la función para añadir reviews a las primeras 100 filas. Quitar
# los comentarios para añadirlo
# add_reviews_to_ratings()

# Mostrar las primeras 100 filas de ratings_df con las nuevas columnas
# print(ratings_df.head(100))

#Hemos generado una GAN con el objetivo de generar reviews, 
# aunque nos sale vacio los ejemplos generados,
#dejo el código para tomarlo como ejemplo
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define el generador
def build_generator(latent_dim, output_shape):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_shape, activation='tanh'))
    return model

# Define el discriminador
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))  # Añade una capa de Flatten
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define y compila la GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

# Entrenamiento de la GAN
def train_gan(generator, discriminator, gan, X_train, latent_dim, epochs=10000, batch_size=64):
    for epoch in range(epochs):
        # Entrenar discriminador
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_reviews = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_reviews = generator.predict(noise)
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_reviews, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_reviews, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Entrenar generador
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Imprimir progreso
        print(f"{epoch} [D loss: {d_loss} | G loss: {g_loss}]")

# Selecciona las reviews reales para entrenar la GAN
selected_reviews = ratings_df['overview'].dropna()
selected_reviews = selected_reviews[:100]  # Selecciona un número de reviews para entrenar

# Tokeniza y convierte las reviews a secuencias numéricas
tokenizer = Tokenizer()
tokenizer.fit_on_texts(selected_reviews)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(selected_reviews)
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Tamaño de la dimensión latente para la GAN
latent_dim = 200
output_shape = max_sequence_length  # Ajusta la salida al máximo de la secuencia

# Construye y compila el generador y el discriminador
generator = build_generator(latent_dim, output_shape)
generator.compile(loss='binary_crossentropy', optimizer='adam')

discriminator = build_discriminator(input_shape=(output_shape,))
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Entrenamiento de la GAN
train_gan(generator, discriminator, gan, padded_sequences, latent_dim, epochs=400, batch_size=64)

# Generación de ejemplos de reviews
def generate_reviews(generator, latent_dim, num_samples=5):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_reviews = generator.predict(noise)
    generated_reviews_text = []

    for sequence in generated_reviews:
        # Obtener las palabras correspondientes a las probabilidades generadas
        generated_words = [tokenizer.index_word.get(np.argmax(prob), "") for prob in sequence]
        generated_review_text = " ".join([word for word in generated_words if word])
        generated_reviews_text.append(generated_review_text)

    return generated_reviews_text

# Imprimir ejemplos generados
generated_reviews_text = generate_reviews(generator, latent_dim, num_samples=5)
for i, review_text in enumerate(generated_reviews_text):
    print(f"Ejemplo generado {i+1}: {review_text}")


#Obtenemos textos vacíos...
