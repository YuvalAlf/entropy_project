# import math
# from abc import ABC
# from dataclasses import dataclass
# from typing import List
#
# from entropy.entropy_vec import EntropyVec
# from utils.plotting_utils import plot_horizontal
#
#
# class DrawnCalculation(ABC):
#     def draw(self, vector1: List[float], vector2: List[float], color: str, label: str, line_style: str) -> None:
#         pass
#
#
# @dataclass
# class EntropyCalculation(DrawnCalculation):
#     max_x: int
#
#     def draw(self, vector1: List[float], vector2: List[float], color: str, label: str, line_style: str) -> None:
#         x_lims = (0, self.max_x)
#         max_entropy_value = math.log(len(vector1))
#         entropy_vector1 = EntropyVec(vector1)
#         entropy_vector2 = EntropyVec(vector2)
#         plot_horizontal(x_lims, max_entropy_value, color='black', linestyle='dashed', alpha=0.8,
#                         label='Max Entropy Value')
#         plot_horizontal(x_lims, entropy_vec1.average_with(entropy_vec2).entropy(), color='gray', linestyle='dashdot',
#                         alpha=0.8, label='Average Vector Entropy')
#         plot_horizontal(x_lims, entropy_vector1.entropy(), color='deeppink', linestyle='dotted', alpha=0.8,
#                         label=f'Entropy of {distribution1_name}')
#         plot_horizontal(x_lims, entropy_vector2.entropy(), color='teal', linestyle='dotted', alpha=0.8,
#                         label=f'Entropy of {distribution2_name}')
#
