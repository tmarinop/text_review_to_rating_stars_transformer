from nltk.corpus import stopwords

useful_words = {"shouldn't", "wasn", "needn't", "hasn't", "isn't", "just", "aren", "aren't", "couldn't", "wouldn't",
                "haven't", "doesn", "weren", "don", "mustn't", "more", "not", "didn't", "hadn", "weren't", "don't",
                "doesn't", "hasn", "very", "won", "wouldn", "most", "haven", "didn", "against", "won't", "needn",
                "wasn't"}

if __name__ == '__main__':
    stop_words = set(stopwords.words('english')) - useful_words
    dataFile = open('../data/stopwords.txt', 'w')
    count = 0
    for word in stop_words:
        if count > 0:
            dataFile.write(',')
        dataFile.write('"')
        dataFile.write(word)
        dataFile.write('"')
        count += 1
