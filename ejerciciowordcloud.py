from PIL import Image
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Descarga de recursos adicionales
nltk.download('punkt')
nltk.download('stopwords')

# Texto proporcionado
texto_alicia = "Alicia estaba comenzando a cansarse de estar sentada junto a su hermana a la orilla del río, sin hacer nada: una o dos veces había echado un vistazo al libro que su hermana leía, pero no tenía dibujos ni diálogos en él, '¿y de qué sirve un libro', pensó Alicia 'sin dibujos ni diálogos?'"

# Tokenización de palabras
palabras_alicia = nltk.word_tokenize(texto_alicia)

# Eliminación de stopwords
stop_words = set(stopwords.words('spanish'))
palabras_filtradas = [palabra.lower() for palabra in palabras_alicia if palabra.isalnum() and palabra.lower() not in stop_words]

# Creación de un objeto WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(palabras_filtradas))

# Visualización de la WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

