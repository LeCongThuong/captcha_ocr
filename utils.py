from pathlib import Path
import os
from typing import List, Set
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_dataset(path_to_images: str) -> (List[str], List[str], Set, int):
    """
    create dataset for image_dir
    :param path_to_images: path to image dir
    :return:
        image_path_list: image path list
        labels: label path list
        set: characters set
        int: max_length of all labels
    """
    img_dir_path = Path(path_to_images)
    image_path_list = sorted(list(map(str, list(img_dir_path.glob("*.png")))))
    labels = sorted(img_path.split(os.path.sep)[-1].split('.')[0] for img_path in image_path_list)
    character_set = set(character for label in labels for character in label)
    max_length = max([len(label) for label in labels])
    return image_path_list, labels, character_set, max_length


def create_dict_between_char_and_num(characters_set: Set):
    char_to_num = layers.experimental.preprocessing.StringLookup(max_token=None, num_oov_indices=0,
                                                                 vocabulary=list(characters_set))
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    return char_to_num, num_to_char


def split_data(images_path_list, labels, train_size=0.8, shuffle_data=True):
    size = len(images_path_list)

    indices = np.arange(size)

    if shuffle_data:
        np.random.shuffle(indices)

    num_train = train_size * size
    train_data = images_path_list[indices[:num_train]]
    train_label = labels[indices[:num_train]]
    val_data = images_path_list[indices[num_train:]]
    val_label = layers[indices[num_train:]]

    return train_data, train_label, val_data, val_label

def encode_single_sample(image_path, label):
    img = tf.i

