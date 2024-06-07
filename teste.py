from sklearn.feature_extraction.text import TfidfVectorizer

text = [
    "o ladrão foi visto roubando um banco",
    "depois de roubar o banco, o ladrão saiu correndo",
    "o ladrão foi visto descansando no banco da praça" 
]

vectorizer = TfidfVectorizer()
vectorizer.fit(text)