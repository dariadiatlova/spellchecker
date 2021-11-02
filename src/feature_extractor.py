import numpy as np
import pandas as pd

from tqdm import tqdm
from spylls.hunspell import Dictionary
from catboost import CatBoostClassifier
from textdistance import Levenshtein, DamerauLevenshtein, JaroWinkler, Jaro, Hamming


class FeatureExtractor:
    def __init__(self):
        self._edit_distance = Levenshtein()
        self.damerau_levenshtein = DamerauLevenshtein()
        self.jaro_winkler = JaroWinkler()
        self.jaro = Jaro()
        self.hamming = Hamming()
        self.distances = [self._edit_distance, self.damerau_levenshtein, self.hamming, self.jaro_winkler, self.jaro]
        self.dictionary = Dictionary.from_files("en_US")
        self.misspelled_words = None
        self.target_words = None
        self.positive_feature_vectors = None
        self.negative_feature_vectors = None
        self.classifier = None

    def _initialize_model(self, model_path: str):
        self.classifier = CatBoostClassifier()
        self.classifier.load_model(model_path)

    def predict(self, word: str, model_path: str, suggest_k_words: int):
        # correct spelling case
        if self.dictionary.lookup(word):
            return "Good job! The spelling is correct!"

        suggestions = np.array(list(self.dictionary.suggest(word)))
        # unknown word case
        if len(suggestions) < 1:
            return word

        # suggest suggest_k_words
        self._initialize_model(model_path)
        feature_vectors = np.stack([self._get_features_to_rank(word, suggested_word) for suggested_word in suggestions])
        suggested_words = suggestions[self._multirank_catboost(feature_vectors)]
        suggested_string = str(suggested_words[0])
        for i in range(1, suggest_k_words):
            suggested_string = suggested_string + ", " + str(suggested_words[i])
        return f"You've probably meant: {suggested_string}?"

    def _get_features_to_rank(self, word_one: str, word_two: str):
        return [distance(word_one, word_two) for distance in self.distances]

    @staticmethod
    def _read_df(df):
        return np.array(df[0]), np.array(df[1])

    @staticmethod
    def _suggestion_check(suggestions, misspelled_word):
        if len(suggestions) < 1:
            print(f"Didn't find any similar words for '{misspelled_word}' word.")
            return np.array(["unknown"])
        return suggestions

    def validate_vanilla_hunspell(self, df):
        self.misspelled_words, self.target_words = self._read_df(df)
        results = np.vstack([self.misspelled_words, np.empty_like(self.misspelled_words, dtype=str), self.target_words])
        for i, misspelled_word, target_word in zip(np.arange(df.shape[0]), self.misspelled_words, self.target_words):
            suggestions = np.array(list(self.dictionary.suggest(misspelled_word)))
            if len(suggestions) >= 1:
                results[1][i] = suggestions[0]
            else:
                results[1][i] = misspelled_word
        accuracy = np.where(results[1, :] == results[2, :])[0].shape[0] / df.shape[0]
        print(f"Vanilla hunspell accuracy: {accuracy}")

    def _rank_catboost(self, feature_vectors):
        return np.argmax(self.classifier.predict_proba(feature_vectors)[:, 1])

    def validate(self, df):
        self.misspelled_words, self.target_words = self._read_df(df)
        results = np.vstack([self.misspelled_words, np.empty_like(self.misspelled_words, dtype=str), self.target_words])

        for i, misspelled_word, target_word in zip(np.arange(df.shape[0]), self.misspelled_words, self.target_words):
            suggestions = np.array(list(self.dictionary.suggest(misspelled_word)))
            suggestions = self._suggestion_check(suggestions, misspelled_word)
            feature_vectors = np.stack([self._get_features_to_rank(misspelled_word, word) for word in suggestions])
            results[1][i] = suggestions[self._rank_catboost(feature_vectors)]

        accuracy = np.where(results[1, :] == results[2, :])[0].shape[0] / df.shape[0]
        print(f"Accuracy: {accuracy}")

        return results

    def _multirank_catboost(self, feature_vectors):
        top_word_idx = np.stack([np.arange(len(feature_vectors)), -self.classifier.predict_proba(feature_vectors)[:, 1]])
        return top_word_idx[1, :].argsort()

    def validate_accuracy_at_k(self, df, k):
        self.misspelled_words, self.target_words = self._read_df(df)
        accuracy = 0

        for i, misspelled_word, target_word in zip(np.arange(df.shape[0]), self.misspelled_words, self.target_words):
            suggestions = np.array(list(self.dictionary.suggest(misspelled_word)))
            suggestions = self._suggestion_check(suggestions, misspelled_word)
            feature_vectors = np.stack([self._get_features_to_rank(misspelled_word, word) for word in suggestions])
            suggested_words = suggestions[self._multirank_catboost(feature_vectors)[:k]]
            if target_word in suggested_words:
                accuracy += 1
        accuracy /= df.shape[0]

        print(f"Accuracy: @{k}: {accuracy}.")

    def _train_catboost(self) -> CatBoostClassifier:
        train_data = np.vstack([self.positive_feature_vectors, self.negative_feature_vectors])
        train_labels = np.concatenate([np.ones(self.positive_feature_vectors.shape[0]),
                                       np.zeros(self.negative_feature_vectors.shape[0])])
        assert len(train_data) == len(train_labels), "Dimensions of train features and labels does not match!"
        self.classifier = CatBoostClassifier(iterations=25, learning_rate=1, depth=5, verbose=False)
        self.classifier.fit(train_data, train_labels)
        return self.classifier

    def train(self, df: pd.DataFrame, train_size: int, save_model_path: str = None) -> CatBoostClassifier:
        df = df.sample(n=train_size)
        self.misspelled_words = np.array(df["wrong"])
        self.target_words = np.array(df["correct"])

        self.positive_feature_vectors = np.zeros((df.shape[0], len(self.distances)), dtype=np.float32)
        self.negative_feature_vectors = np.zeros_like(self.positive_feature_vectors, dtype=np.float32)

        for i, misspelled_word, target_word in zip(tqdm(np.arange(df.shape[0]), total=df.shape[0]),
                                                   self.misspelled_words, self.target_words):
            suggestions = np.array(list(self.dictionary.suggest(misspelled_word)))
            if len(suggestions) >= 1:
                wrong_predicted_word = np.random.choice(suggestions)
            else:
                wrong_predicted_word = "random"
                print(f"Did not find wrong suggestion for {misspelled_word} word.")

            self.positive_feature_vectors[i] = self._get_features_to_rank(misspelled_word, target_word)
            self.negative_feature_vectors[i] = self._get_features_to_rank(misspelled_word, wrong_predicted_word)

        classifier = self._train_catboost()
        if save_model_path:
            self.classifier.save_model(save_model_path)
        return classifier
