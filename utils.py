from pathlib import Path
import os
from typing import List, Set
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_dict_between_char_and_num(characters_set: Set):
    char_to_num = layers.experimental.preprocessing.StringLookup(max_token=None, num_oov_indices=0,
                                                                 vocabulary=list(characters_set))
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    return char_to_num, num_to_char


