import matplotlib.pyplot as plt
from nltk import FreqDist
from wordcloud import WordCloud
from features.process_text import process_text


# https://www.datacamp.com/community/tutorials/wordcloud-python


def visualize_words(text, tokenized_stop_words, cleanup=False):
    """Generate both wordcloud and frequency distribution visualizations when passed a list of text strings and a
    list of stopwords to remove. Both visualizations make use of wordlcoud's ability to process text, so that the
    word statistics correspond. Show these figures side by side."""
    plt.figure(figsize=(18, 6), dpi=80)

    processed_text = process_text(text, tokenized_stop_words, clean=cleanup)  # Tokenize & remove stopwords
    fdist = FreqDist(processed_text)

    plt.subplot(1, 2, 1)
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(fdist)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    fdist.plot(30, cumulative=False)
    plt.show()
    print(fdist.most_common(30))

# plot = flora_data_frame['length'].hist(by=flora_data_frame['classification'])
# plt.show()
