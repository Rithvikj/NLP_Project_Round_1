import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import FreqDist, pos_tag
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
import numpy as np

# In this the book used is 'COMPUTER NETWORKS' by ANDREW S. TANENBAUM and DAVID J. WETHERALL
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

file = open('book1.txt', 'r', encoding='utf-8')
text = file.read()

file.close()
text = re.sub(r'[^\x00-\x7f]', " ", text)# Removing non-ASCII characters
text = re.sub( "\n[0-9]+\n" , "", text)# Removing page numbers
text = re.sub( "\nCHAP. [0-9]+\n" , "", text)# Removing 'CHAP. 1' like words
text = re.sub( "\nSEC. [0-9]+.[0-9]+\n" , "", text)# Removing 'SEC. 1.1' like words
text = re.sub( "[Ff]ig.[ ][0-9]+-[0-9]+" , "", text)# Removing 'Figure' keyword
text = re.sub( "[Ff]igure[ ][0-9]+-[0-9]+" , "", text)# Removing 'Fig. ' keyword
text = re.sub( "\n" , " ", text)# Removing linebreak
text = re.sub("""[^\w\s]""", " ", text)# Removing all punctuation marks

text = ' '.join(text.split()) # Removing extra spaces

# Converting all characters to lower case
text2 = ""

for char in text:
    text2 += char.lower()

print(text2[:20000])

# Tokenizing the words
tokenized_words = word_tokenize(text2)
for i in range(0,40):
    print(tokenized_words[i])


# Plotting frequency distribution of tokens with stopwords
freq_distribution_1 = FreqDist(tokenized_words)
freq_distribution_1.plot(20, title='Frequency Distribution of Tokens With Stopwords')  # this will give 20 most frequently occurring characters


# Creating word cloud of tokens
WC = WordCloud(width = 900, height = 900, min_font_size= 12, background_color="white", max_words=100).generate(text2)
plt.figure(figsize = (12,8))
plt.imshow(WC)
plt.axis('off')
plt.title('Word Cloud With Stopwords')
plt.show()

# Removing stopwords
Stopwords = set(stopwords.words('english'))
tokenized_words_2 = []
for word in tokenized_words:
    if word not in Stopwords:
        tokenized_words_2.append(word)

text3 = ' '.join(tokenized_words_2)
print(text3[:20000])

# Plotting frequency distribution of tokens without stopwords
freq_distribution_2 = FreqDist(tokenized_words_2)
freq_distribution_2.plot(20, title='Frequency Distribution of Tokens Without Stopwords')

# Creating word cloud of tokens without stopwords
WC2 = WordCloud(width = 900, height = 900, min_font_size= 12, background_color="white", max_words=100, stopwords=Stopwords).generate(text3)
plt.figure(figsize = (12,8))
plt.imshow(WC2)
plt.axis('off')
plt.title('Word Cloud Without Stopwords')
plt.show()

# Performing POS tagging and plotting its frequency distribution
tagged_words = pos_tag(tokenized_words_2)
for i in range(0,40):
    print(tagged_words[i])


tag_freq = [word[1] for word in tagged_words]
fd = FreqDist(tag_freq)
fd.plot(20, title='Top 20 most common tags')



# Creating Frequency distribution of word length of the tokens
word_len=[]
for word in tokenized_words_2:
    word_len.append(len(word))

bin_size=np.linspace(0, 16)
plt.hist(word_len, bins=bin_size)
plt.xlabel('word length')
plt.ylabel('word length Frequency')
plt.title('Frequency Distribution of word length')
plt.show()