import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Set
from pathlib import Path
import os
from utils import create_dict_between_char_and_num


class CaptchaDataset:
    def __init__(self, config):
        self.config = config
        self.image_path_list, self.labels, self.character_set, self.max_length = self.create_image_lists()
        self.char_to_num, self.num_to_char = create_dict_between_char_and_num(self.character_set)

    def create_dataset(self):
        train_data, train_label, val_data, val_label = self.split_data()
        train_dataset = self.create_data_pipline(train_data, train_label)
        val_dataset = self.create_data_pipline(val_data, val_label)
        return train_dataset, val_dataset

    def create_data_pipline(self, image_data, image_label):
        captcha_dataset = tf.data.Dataset.from_tensor_slices((image_data, image_label))
        captcha_dataset = captcha_dataset.map(self.encode_single_sample,
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE
                                              ).batch(self.config.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return captcha_dataset

    def create_image_lists(self) -> (List[str], List[str], Set, int):
        """
        create dataset for image_dir
        :return:
            image_path_list: image path list
            labels: label path list
            set: characters set
            int: max_length of all labels
        """
        img_dir_path = Path(self.config.path_to_images)
        image_path_list = sorted(list(map(str, list(img_dir_path.glob("*.png")))))
        labels = sorted(img_path.split(os.path.sep)[-1].split('.')[0] for img_path in image_path_list)
        character_set = set(character for label in labels for character in label)
        max_length = max([len(label) for label in labels])
        return image_path_list, labels, character_set, max_length

    def split_data(self):
        size = len(self.image_path_list)

        indices = np.arange(size)

        if self.config.data_shuffle:
            np.random.shuffle(indices)
        num_train = int(self.config.train_size * size)
        train_data = np.array(self.image_path_list)[indices[:num_train]]
        train_label = np.array(self.labels)[indices[:num_train]]
        val_data = np.array(self.image_path_list)[indices[num_train:]]
        val_label = np.array(self.labels)[indices[num_train:]]
        return train_data, train_label, val_data, val_label

    def encode_single_sample(self, image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.config.img_height, self.config.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}
