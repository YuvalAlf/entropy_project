from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, DefaultDict, Tuple

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups

from entropy.entropy_vec import EntropyVec
from utils.core_utils import fst
from utils.functional_utils import map_list


@dataclass
class NewsgroupThemeTokens:
    theme: str
    token_to_count: DefaultDict[str, int]

    @staticmethod
    def unique_tokens(newsgroups: List[NewsgroupThemeTokens], num_tokens: int) -> (List[str], Dict[str, int]):
        unique_tokens = {token for newsgroup in newsgroups for token in newsgroup.token_to_count.keys()}
        top_unique_tokens = sorted(unique_tokens, key=lambda t: sum(n.token_to_count[t] for n in newsgroups), reverse=True)[:num_tokens]
        token_to_location = map_list(fst, enumerate(top_unique_tokens))
        return top_unique_tokens, token_to_location

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

    def probability_vector(self, unique_tokens: List[str]) -> EntropyVec:
        count_vector = [self.token_to_count.get(token, 0) for token in unique_tokens]
        return EntropyVec(count_vector).normalize()

    @staticmethod
    def probability_vectors(num_newsgroups_in_each_theme: int, num_tokens: int) -> List[Tuple[str, EntropyVec]]:
        newsgroups = NewsgroupThemeTokens.fetch(num_newsgroups_in_each_theme)
        unique_tokens, token_to_location = NewsgroupThemeTokens.unique_tokens(newsgroups, num_tokens)
        return [(newsgroup.theme, newsgroup.probability_vector(unique_tokens)) for newsgroup in newsgroups]
