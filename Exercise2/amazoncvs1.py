import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


data = pd.read_csv('/home/luankaio/projetos/nlpProject/Exercise2/AmazonReview.csv')
data.head()

#drop null values
data.dropna(inplace=True)


data.loc[data['Sentiment']<=3,'Sentiment'] = 0
 
#4,5->positive(i.e 1)
data.loc[data['Sentiment']>3,'Sentiment'] = 1

#cleaning stopword from the dataset
stp_words=stopwords.words('english')
def clean_review(review): 
  cleanreview=" ".join(word for word in review.
                       split() if word not in stp_words)
  return cleanreview 
 
data['Review']=data['Review'].apply(clean_review)

data['Sentiment'].value_counts()
