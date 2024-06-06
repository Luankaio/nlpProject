#mostra as palavras que cercam a palavra selecionada
import nltk
from nltk.tokenize import word_tokenize

nltk.download('machado')

corpus_dom_casmurro = nltk.corpus.machado.raw('romance/marm08.txt')

dom_casmurro = nltk.Text(word_tokenize(corpus_dom_casmurro))
dom_casmurro.concordance('Capitu')
print("-----------------")

#palavras que ocorrem no mesmo contexto
dom_casmurro.similar('padre')
print("-----------------")

dom_casmurro.collocations()
