from tensorflow.keras.datasets import imdb
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
# Cargar el dataset de IMDb
# num_words=10000 mantiene las 10,000 palabras más frecuentes en el dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()

# Obtener el índice inverso
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decodificar las reseñas
decoded_reviews = [' '.join([reverse_word_index.get(i - 3, '?') for i in review]) for review in x_train]
word_index = imdb.get_word_index()

# Obtener el índice inverso
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decodificar las reseñas
decoded_reviews = [' '.join([reverse_word_index.get(i - 3, '?') for i in review]) for review in x_train]

# Tokenizar las reseñas decodificadas
tokenized_reviews = [word_tokenize(review) for review in decoded_reviews]

# Entrenar el modelo Word2Vec
modelo = Word2Vec(tokenized_reviews, vector_size=100, window=5, min_count=2, workers=4)

palabras_similares = modelo.wv.most_similar('good', topn=5)
print(palabras_similares)