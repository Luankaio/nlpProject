import nltk

# Lista de palavras
palavras = ["Olá", "mundo", "isso", "é", "um", "exemplo", "de", "nltk.Text"]

# Criando um objeto nltk.Text
texto_nltk = nltk.Text(palavras)

# Exemplo de utilização
print(texto_nltk.concordance("exemplo"))
