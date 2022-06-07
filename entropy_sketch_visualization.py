import json
from random import Random

import matplotlib.pyplot as plt
import pandas as pd

from entropies_bound import generators
from entropy.entropy_sketch import EntropySketch
from entropy.entropy_vec import EntropyVec
from utils.data_frame_aggragator import DataFrameAggragator


def main() -> None:
    output_csv = 'entropy_sketch.csv'
    vector_length = 10000
    prng = Random(222)
    df_aggragator = DataFrameAggragator()
    for generator_name, generator in generators():
        probability_vector = EntropyVec.gen_rand(vector_length, generator)
        probability_vector.show(generator_name)
        # sketch_values = []
        # real_entropy = probability_vector.entropy()
        # for sketch_size in range(10, 1001, 10):
        #     approximated_entropy, sketch = EntropySketch(sketch_size, vector_length, prng).apply(probability_vector)
        #     sketch_values.extend(sketch)
        #     df_aggragator.append_row(generator=generator_name, sketch_size=sketch_size, real_entropy=real_entropy,
        #                              approximated_entropy=approximated_entropy)
        #     print(sketch_size)
        # with open(f'sketch_values_{generator_name}.json', 'w') as file:
        #     json.dump(sketch_values, file, indent=True)
        # pd.Series([value for value in sketch_values if value > -200]).plot.kde(bw_method=0.05)
        # plt.xlim(-200, max(max(sketch_values) + 1, 0))
        # plt.xlabel('Linear Projection Value')
        # plt.ylabel('Density')
        # plt.savefig(f'sketch_{generator_name}_values_kde.png', dpi=300, bbox_inches='tight')
    df = df_aggragator.to_data_frame()
    df.to_csv(output_csv)

    for generator_name in df['generator'].unique():
        sub_df = df[df['generator'] == generator_name]
        plt.figure(figsize=(10, 10))
        plt.scatter(sub_df['sketch_size'], sub_df['approximated_entropy'], clip_on=False, color='r', alpha=0.6, label='Approximated Entropy',linewidths=0.5, s=8)
        plt.plot(sub_df['sketch_size'], sub_df['real_entropy'], clip_on=False, color='g', alpha=0.85, label='Real Entropy', lw=2)

        plt.xlim((0, None))
        # plt.ylim((0, math.ceil(math.log(vector_length))))
        plt.legend(loc='lower right')
        plt.xlabel('Sketch Size')
        plt.ylabel('Entropy')
        plt.savefig(f'sketch_{generator_name}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
