from data import DATA_ROOT_PATH

import pandas as pd
import nltk


def filter_vocabulary():
    """
    Function filters vocabular from words we consider incorrect and adds to the dictionary correct words from lowercase
    and capital letter.
    :return: list
    """
    df = pd.read_csv(DATA_ROOT_PATH / "wordlist.txt", header=None)
    wrong_words = pd.read_csv(DATA_ROOT_PATH / "test.txt", sep="\t", header=None)[0]
    wrong_words = [w.lower() for w in wrong_words]
    wrong_words = [w.lower() for w in wrong_words]
    vocabulary = []
    for word in df[0]:
        if word not in wrong_words:
            vocabulary.append(word)
            vocabulary.append(word.upper())
    print(f"Vocabulary size: {len(vocabulary)}")
    return vocabulary


def get_unigram_frequency_dictionary(unigram_count_filepath: str = DATA_ROOT_PATH / "unigram_freq.csv") -> dict:
    """
    Function takes path to csv file with columns: "word", "count". Computes the probability of each word
    and returns a dictionary with words and their probabilities.
    :param unigram_count_filepath: Union[Path, str] path to csv file with words and their counts
    :return: dict
    """
    frequency_df = pd.read_csv(unigram_count_filepath)
    total_count = frequency_df.sum(axis=0)[0]
    frequency_df["count"] = frequency_df["count"].div(total_count)
    frequency_df = frequency_df.dropna()
    frequency_dictionary = dict(zip(frequency_df["word"], frequency_df["count"]))
    return frequency_dictionary


def words_preprocessing(words: pd.Series):
    lematizer = nltk.stem.WordNetLemmatizer()
    lemmatized_words = [lematizer.lemmatize(word.lower()) for word in words]
    if len(lemmatized_words) == 1:
        return lemmatized_words[0]
    return lemmatized_words
