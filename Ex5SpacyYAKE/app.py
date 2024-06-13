import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Carregando o modelo de linguagem do Spacy
nlp = spacy.load('pt_core_news_lg')

# Função para pré-processar e etiquetar o texto com POS tagging
def preprocess_text_with_pos(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ + '_' + token.pos_ for token in doc])

# Carregando os dados
data = pd.read_csv('spam.csv')

# Pré-processamento dos textos com POS tagging
data['Processed_Text'] = data['Message'].apply(preprocess_text_with_pos)

def adjust_weights_based_on_pos(weights, pos_tags):
    adjusted_weights = []
    for weight, pos_tag in zip(weights, pos_tags):
        if pos_tag.startswith("N"):  # Substantivo
            adjusted_weights.append(weight * 1.5)  # Aumentar peso
        elif pos_tag.startswith("V"):  # Verbo
            adjusted_weights.append(weight * 1.2)  # Aumentar peso
        else:
            adjusted_weights.append(weight)  # Manter o mesmo peso para outras classes
    return adjusted_weights


# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data['Processed_Text'], data['Category'], test_size=0.3, random_state=42)

# Vetorização com TF-IDF ponderado pelo POS
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Treinamento do modelo de classificação (por exemplo, RandomForest)
clf = RandomForestClassifier()
clf.fit(X_train_tfidf, y_train)

# Avaliação do modelo
accuracy = clf.score(X_test_tfidf, y_test)
print("A precisão do modelo é:", accuracy)
