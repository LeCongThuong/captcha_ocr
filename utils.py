from pathlib import Path
import os
from typing import List, Set
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
from bunch import Bunch
import os
import argparse


def create_dict_between_char_and_num(characters_set: Set):
    char_to_num = layers.experimental.preprocessing.StringLookup(mask_token=None, num_oov_indices=0,
                                                                 vocabulary=list(characters_set))
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    return char_to_num, num_to_char


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args



