import argparse
import json

from data import DATA_ROOT_PATH
from src.feature_extractor import FeatureExtractor


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-w', '--word',
                        type=str,
                        help='Word to check spelling.')

    parser.add_argument('-i', '--input_config_path',
                        type=str,
                        default=DATA_ROOT_PATH.parent / 'config.json',
                        help='Path to json configuration file.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(args.input_config_path))

    spellchecker = FeatureExtractor()
    print(spellchecker.predict(args.word, config["model_path"], config["suggest_k_words"]))
