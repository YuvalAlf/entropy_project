from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups

from utils.core_utils import fst, snd
from utils.functional_utils import map_list


@dataclass
class NewsgroupThemeTokens:
    theme: str
    token_to_count: Dict[str, int]

    @staticmethod
    def unique_tokens(newsgroups: NewsgroupThemeTokens) -> (List[str], Dict[str, int]):
        unique_tokens = sorted({token for newsgroup in newsgroups for token in newsgroup.token_to_count.keys()})
        token_to_location = map_list(fst, enumerate(unique_tokens))
        return unique_tokens, token_to_location

    @staticmethod
    def fetch(num_newsgroups: int) -> List[NewsgroupThemeTokens]:
        newsgroups_count = defaultdict(int)
        newsgroups = fetch_20newsgroups(subset='train')
        target_to_number_of_tokens = defaultdict(lambda: defaultdict(int))
        stop_words = stopwords.words('english')
        for target_num, data in zip(newsgroups.target, newsgroups.data):
            if newsgroups_count[target_num] < num_newsgroups:
                newsgroups_count[target_num] += 1
                theme = newsgroups.target_names[target_num]
                for token in word_tokenize(data):
                    if token not in stop_words:
                        lower_token = token.lower()
                        if all('a' <= ch <= 'z' for ch in token):
                            target_to_number_of_tokens[theme][lower_token] += 1
        return [NewsgroupThemeTokens(theme, token_to_count) for theme, token_to_count in sorted(target_to_number_of_tokens.items(), key=fst)]

    def probability_vector(self, unique_tokens: List[str]) -> np.ndarray:
        count_vector = np.array([self.token_to_count.get(token, 0) for token in unique_tokens])
        return count_vector / np.sum(count_vector)


def calc_frequency(current_tokens: List[str], num_tokens: int, token_to_location: Dict[str, int]) -> np.ndarray:
    counter = np.zeros(num_tokens)
    for token, times in Counter(current_tokens).items():
        if token in token_to_location:
            counter[token] = times
    return counter / counter.sum()


def calc_entropy(probability_vector: np.ndarray) -> float:
    return sum(-p*math.log(p) for p in probability_vector if p > 0)


def average_probabilities(list1: List[float], list2: List[float]) -> List[float]:
    return [value1 + value2 for value1, value2 in zip(list1, list2)]


def calc_entropy_matrix(probabilities: List[np.ndarray]) -> np.ndarray:
    size = len(probabilities)
    matrix = np.empty((size, size))
    for i, prob_vec1 in enumerate(probabilities):
        for j, prob_vec2 in enumerate(probabilities):
            matrix[i, j] = calc_entropy((prob_vec1 + prob_vec2) / 2)
    return matrix


def main() -> None:
    newsgroups = NewsgroupThemeTokens.fetch(100)
    unique_tokens, token_to_location = NewsgroupThemeTokens.unique_tokens(newsgroups)
    themes = [newsgroup.theme for newsgroup in newsgroups]
    probability_vectors = [newsgroup.probability_vector(unique_tokens) for newsgroup in newsgroups]
    print(calc_entropy(sum(probability_vectors) / len(probability_vectors)))

    sum_vec = sum(probability_vectors) / len(probability_vectors)
    for token, prob in sorted(zip(unique_tokens, probability_vectors[0]), key=snd):
        print(f'{token}\t{prob:.12f}%')

    entropy_matrix = calc_entropy_matrix(probability_vectors)
    print(f'Max Entropy = {math.log(len(unique_tokens))}')
    plt.figure(figsize=(12, 12))
    sns.heatmap(square=True, linewidths=1,
                data=entropy_matrix, xticklabels=themes, yticklabels=themes, cmap='summer',
                annot=True, fmt='.2f', annot_kws={'color': 'black'})
    plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
    os.system('heatmap.png')


if __name__ == '__main__':
    main()
