import matplotlib.pyplot as plt
from nltk import FreqDist
from wordcloud import WordCloud
from features.process_text import process_text
import numpy as np

def blue_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    """This function recolors the word cloud. Colours selected in a very naive manner."""
    return "hsl(211, 53%%, %d%%)" % np.random.randint(50, 90)  # https://amueller.github.io/word_cloud/auto_examples/a_new_hope.html

def red_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    """This function recolors the word cloud. Colours selected in a very naive manner."""
    return "hsl(359, 82%%, %d%%)" % np.random.randint(50, 90)  # https://amueller.github.io/word_cloud/auto_examples/a_new_hope.html

def yellow_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    """This function recolors the word cloud. Colours selected in a very naive manner."""
    return "hsl(30, 61%%, %d%%)" % np.random.randint(50, 90)  # https://amueller.github.io/word_cloud/auto_examples/a_new_hope.html

def purple_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    """This function recolors the word cloud. Colours selected in a very naive manner."""
    return "hsl(175, 36%%, %d%%)" % np.random.randint(50, 90)  # https://amueller.github.io/word_cloud/auto_examples/a_new_hope.html


def visualize_words(text, tokenized_stop_words, cleanup=False, color=None): # https://www.datacamp.com/community/tutorials/wordcloud-python
    """Generate both wordcloud and frequency distribution visualizations when passed a list of text strings and a
    list of stopwords to remove. Both visualizations make use of wordlcoud's ability to process text, so that the
    word statistics correspond. Show these figures side by side."""
    plt.figure(figsize=(18, 6), dpi=80)

    processed_text = process_text(text, tokenized_stop_words, clean=cleanup)  # Tokenize & remove stopwords
    fdist = FreqDist(processed_text)

    plt.subplot(1, 2, 1)
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(fdist)

    if color == "blue":
        wordcloud = wordcloud.recolor(color_func=blue_color_func)
    elif color == "red":
        wordcloud = wordcloud.recolor(color_func=red_color_func)
    elif color == "yellow":
        wordcloud = wordcloud.recolor(color_func=yellow_color_func)
    elif color == "purple":
        wordcloud = wordcloud.recolor(color_func=purple_color_func)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    fdist.plot(30, cumulative=False)
    plt.show()
    print(fdist.most_common(30))

#    return wordcloud

# plot = flora_data_frame['length'].hist(by=flora_data_frame['classification'])
# plt.show()
