import nltk
nltk.download('machado')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import re
from nltk.draw.dispersion import dispersion_plot
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')


def pre_processamento(texto):
  
    # seleciona apenas letras e coloca todas em minúsculo 
    letras_min =  re.findall(r'\b[A-zÀ-úü]+\b', texto.lower())

    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stop = set(stopwords)
    sem_stopwords = [w for w in letras_min if w not in stop]

    # juntando os tokens novamente em formato de texto
    texto_limpo = " ".join(sem_stopwords)

    return texto_limpo


# corpus dom casmurro
corpus_dom_casmurro = nltk.corpus.machado.raw('romance/marm08.txt')

# pre processamento
texto = pre_processamento(corpus_dom_casmurro)

# tokenizando 
tokens = word_tokenize(texto)

pp = ['capitu', 'mãe', 'olhos', 'seminário', 'amor', 'bentinho', 'escobar','ezequiel']

plt.figure(figsize=(15, 10))

my_plot = dispersion_plot(tokens, pp)

if [label.get_text() for label in my_plot.get_yticklabels()] != reversed(pp):
    my_plot.set_yticks(list(range(len(pp))), reversed(pp))

plt.show()


