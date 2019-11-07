import pandas as pd
import wikipedia
import glob


# These functions are to assist in the extraction of Wikipedia pages for annotation.

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
    str(page) == "<WikipediaPage 'Cirsium arvense'>", 'This functions expects a WikipediaPage as input'
    page_sections = page.sections
    parsed_page = [(page_section_name, page.section(page_section_name), page.title, 'wikipedia') for page_section_name in
                   page_sections]
    wiki_data = pd.DataFrame(parsed_page, columns=['classification', 'text', 'species', 'dataset_name'])
    page_summary = pd.Series(["Summary", page.summary, page.title, 'wikipedia'], index=wiki_data.columns)
    wiki_data = wiki_data.append(page_summary, ignore_index=True)  # Add the page summary
    return wiki_data


def make_wiki_data_frame(species_list):
    """Run wiki extraction functions on a species list.
    :param species_list: A list of strings with species names to use as queries for Wikipedia API
    :return:
        Write the resulting pandas data frame to a csv. The csv is given the name of the last species in the species list.
    """
    wiki_pages = [extract_wiki_page(species_name) for species_name in species_list]
    wiki_data = [extract_wiki_page_data(page) for page in wiki_pages if page != None]  # Remove pages not returned
    wiki_data = pd.concat(wiki_data, ignore_index=True)
    wiki_data.to_csv(path_or_buf="../../data/interim/wiki_data/" + species_list[-1] + ".csv")


def batch_download(species_list, batch_start=1):
    """Batch the download of Wikipedia download.
    :param species_list: The full species list to be batched.
    :param batch_start: A place to begin, if download halted in the middle of a batch because of a timeout.
    :return: Run the make_wiki_data_frame, which saves batch data as csv files.
    """
    print("Batching species list...")
    batches = [species_list[i: i + 10] for i in range(0, len(species_list), 10)]
    for num, batch in enumerate(batches, start=batch_start):
        print("Working on batch number " + str(num))
        make_wiki_data_frame(batch)


def main():
    """
    Develop a csv to store the page data from Wikipedia based on a species list from the Illustrated Flora of BC
    (species column from the flora_data_frame)
    :return: combined.csv
    """
    flora_data_frame = pd.read_csv("../../data/processed/flora_data_frame_full.csv", index_col=0)
    species_list = list(flora_data_frame.species.unique())
    species_list = [species for species in species_list if str(species) != 'nan']  # Remove nan species names
    batch_download(species_list)

    all_filenames = [name for name in glob.glob('../../data/interim/wiki_data/*.{}'.format('csv'))]
    combined_csv = pd.concat([pd.read_csv(file) for file in all_filenames], ignore_index=True)
    combined_csv = combined_csv.dropna(subset=['text']) # a little data cleaning. Remove rows with text = NaN
    combined_csv.to_csv("../../data/processed/combined_7-nov-2019.csv", index=False, encoding='utf-8-sig') # export to csv


if __name__ == '__main__':
    main()
