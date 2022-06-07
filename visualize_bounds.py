import math
from math import log

import matplotlib.pyplot as plt
import pandas as pd

from entropies_bound import TOP_N_ENTRY, DISTRIBUTION_1_ENTRY, DISTRIBUTION_2_ENTRY, AVERAGE_ENTROPY_ENTRY, \
    VEC_LENGTH_ENTRY, LOWER_BOUND_VEC_1_ENTRY, LOWER_BOUND_VEC_2_ENTRY, ENTROPY_1_ENTRY, ENTROPY_2_ENTRY, \
    UPPER_BOUND_VEC_2_ENTRY, UPPER_BOUND_VEC_1_ENTRY


def main() -> None:
    df = pd.read_csv('result.csv')
    for distribution1 in df[DISTRIBUTION_1_ENTRY].unique():
        for distribution2 in df[df[DISTRIBUTION_1_ENTRY] == distribution1][DISTRIBUTION_2_ENTRY].unique():
            sub_df = df[(df[DISTRIBUTION_1_ENTRY] == distribution1) & (df[DISTRIBUTION_2_ENTRY] == distribution2)]
            max_entropy_value = log(sub_df[VEC_LENGTH_ENTRY].mean())
            plt.figure(figsize=(10, 10))
            plt.plot(sub_df[TOP_N_ENTRY], sub_df[AVERAGE_ENTROPY_ENTRY], color='green', linestyle='dotted', alpha=0.8,
                     label='Average Vector Entropy')
            plt.plot(sub_df[TOP_N_ENTRY], sub_df[ENTROPY_1_ENTRY], color='magenta', linestyle='dotted', alpha=0.8,
                     label=f'Entropy of {distribution1}')
            plt.plot(sub_df[TOP_N_ENTRY], sub_df[ENTROPY_2_ENTRY], color='orange', linestyle='dotted', alpha=0.8,
                     label=f'Entropy of {distribution2}')
            plt.plot(sub_df[TOP_N_ENTRY], sub_df[LOWER_BOUND_VEC_1_ENTRY], color='blue', alpha=0.8,
                     label=f'Lower Bound: {distribution1}')
            plt.plot(sub_df[TOP_N_ENTRY], sub_df[LOWER_BOUND_VEC_2_ENTRY], color='red', alpha=0.8,
                     label=f'Lower Bound: {distribution2}')
            plt.plot(sub_df[TOP_N_ENTRY], sub_df[UPPER_BOUND_VEC_1_ENTRY], color='purple', alpha=0.8,
                     label=f'Upper Bound: {distribution1}')
            plt.plot(sub_df[TOP_N_ENTRY], sub_df[UPPER_BOUND_VEC_2_ENTRY], color='brown', alpha=0.8,
                     label=f'Upper Bound: {distribution2}')
            plt.plot(sub_df[TOP_N_ENTRY], [max_entropy_value] * len(sub_df[TOP_N_ENTRY]), color='gray',
                     linestyle='dashed', alpha=0.8, label='Max Entropy Value')
            plt.xlim((0, None))
            plt.ylim((None, math.ceil(max_entropy_value)))
            plt.legend(loc='lower right')
            plt.xlabel('Number of Top Values Transmitted')
            plt.ylabel('Entropy')
            plt.savefig(f'results\\{distribution1}_{distribution2}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
