import nltk
import timeit
import math
nltk.download('machado')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import re
nltk.download('punkt')
nltk.download('stopwords')

tempo = timeit.timeit('math.sqrt(25)', setup='import math', number=1000000)
def pre_processamento(texto):
  
    # seleciona apenas letras e coloca todas em minúsculo 
    letras_min =  re.findall(r'\b[A-zÀ-úü]+\b', texto.lower())

    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stop = set(stopwords)
    sem_stopwords = [w.lower() for w in letras_min if w.lower() not in stop]

    # juntando os tokens novamente em formato de texto
    texto_limpo = " ".join(sem_stopwords)

    return texto_limpo



# corpus dom casmurro
corpus_dom_casmurro = nltk.corpus.machado.raw('romance/marm08.txt')

# pre processamento
texto = pre_processamento(corpus_dom_casmurro)

# tokenizando 
tokens = nltk.Text(word_tokenize(corpus_dom_casmurro))
tokens.collocations()
# contagem de frequencia
fd = FreqDist(tokens)
print("20 palavras mais frequentes:")
print(fd.most_common(20))

# plot
import matplotlib.pyplot as plt
plt.figure(figsize = (13, 8))
fd.plot(30, title = "Frequência de Palavras")

print(tempo)