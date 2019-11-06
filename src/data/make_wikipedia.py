import pandas as pd
import wikipedia
# These functions are to assist in extract of Wikipedia pages for annotation.

def extract_wiki_page(species_name):
    """Uses the Wikipedia Python library & API to extract Wikipedia articles as page objects.
    :param:
        A species name in string format.
    :return:
        WikipediaPage objects. Returns None if no Wikipedia page is found.
    Notes:
        The Wikipedia library tries to match pages greedily, and returns pages that are not exact matches."""
    assert type(species_name) == str, 'Species name not string format'
    try:
        page = wikipedia.page(species_name)
    except:
        page = None
    return page


def extract_wiki_page_data(page):
    """Get a list of the page sections (i.e., the text
    classifications) & pull the text from these sections. Extract page summary text.
    :param:
        A WikipediaPage object from the Wikipedia library
    :return:

    Put all of this into a dataframe for downstream model usage. Return a pandas dataframe """
    page_sections = page.sections
    parsed_page = [(page_section_name, page.section(page_section_name), page.title) for page_section_name in page_sections]
    wiki_data = pd.DataFrame(parsed_page, columns=['classification', 'text', 'species'])
    page_summary = pd.Series(["Summary", page.summary, page.title], index=wiki_data.columns)
    wiki_data = wiki_data.append(page_summary, ignore_index=True)  # Add the page summary
    return wiki_data


def make_wiki_data_frame(species_list):
    """

    :param species_list:
    :return:

    """
    wiki_pages = [extract_wiki_page(species_name) for species_name in species_list]
    wiki_data = [extract_wiki_page_data(page) for page in wiki_pages if page != None]  # Remove pages not returned
    wiki_data = pd.concat(wiki_data, ignore_index=True)
    wiki_data.to_csv(path_or_buf="../data/interim/wiki_data/" + species_list[-1] + ".csv")


def main():
    batches = [species_list[i: i + 10] for i in range(0, len(species_list), 10)]
    for num, batch in enumerate(batches):
        print(num)
        make_wiki_data_frame(batch)

if __name__ == '__main__':

