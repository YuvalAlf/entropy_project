from collections import defaultdict
from typing import Any

import pandas as pd
from more_itertools import all_equal

from utils.core_utils import len_func


class DataFrameAggragator:
    def __init__(self):
        self.dictionary = defaultdict(list)

    def append_row(self, **name_to_value: Any):
        for name, value in name_to_value.items():
            self.dictionary[name].append(value)

    def add_value(self, **name_to_value: Any):
        for name, value in name_to_value.items():
            self.dictionary[name].append(value)
        assert all_equal(map(len_func, self.dictionary.values()))

    def to_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.dictionary)
