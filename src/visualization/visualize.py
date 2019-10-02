from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# https://www.datacamp.com/community/tutorials/wordcloud-python

def visualize_words(text, tokenized_stop_words):
    """Generate both wordcloud and frequency distribution visualizations when passed a list of text strings and a
    list of stopwords to remove. Both visualizations make use of wordlcoud's ability to process text, so that the
    word statistics correspond. Show these figures side by side."""
    plt.figure(figsize=(18, 6), dpi=80)

    wordcloud = WordCloud(stopwords=tokenized_stop_words, collocations=False,
                          background_color="white").generate(text)
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    word_cloud_text = WordCloud(stopwords=tokenized_stop_words, collocations=False).process_text(text)
    fdist = FreqDist(word_cloud_text)
    plt.subplot(1, 2, 2)
    fdist.plot(30, cumulative=False)
    plt.show()
    print(fdist.most_common(30))
