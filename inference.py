import tensorflow as tf
from tensorflow import keras
from utils import get_args, get_config_from_json
import numpy as np
from dataset.captcha_dataset import CaptchaDataset
import matplotlib.pyplot as plt
from tensorflow.keras import layers


def main():
    try:
        args = get_args()
        config, _ = get_config_from_json(args.config)
    except Exception as e:
        print(e)
        print("missing or invalid arguments")
        exit(0)

    model = keras.models.load_model(config.model_save_dir)

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    prediction_model.summary()

    # A utility function to decode the output of the network
    captch_dataset = CaptchaDataset(config)
    train_dataset, val_dataset = captch_dataset.create_dataset()
    char_to_num_file_path = config.model_save_dir + '/char_to_num.txt'
    num_to_char_file_path = config.model_save_dir + '/num_to_char.txt'
    with open(char_to_num_file_path) as f:
        char_to_num_list = f.read().splitlines()

    with open(num_to_char_file_path) as f:
        num_to_char_list = f.read().splitlines()

    captch_dataset.char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num_list)
    captch_dataset.num_to_char = layers.experimental.preprocessing.StringLookup(vocabulary=num_to_char_list)


if __name__ == '__main__':
    main()
