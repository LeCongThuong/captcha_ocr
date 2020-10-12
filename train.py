import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from utils import get_args, get_config_from_json, decode_batch_predictions
from models.ctc_model import CTCModel
from dataset.captcha_dataset import CaptchaDataset


def main():
    try:
        args = get_args()
        config, _ = get_config_from_json(args.config)
    except Exception as e:
        print(e)
        print("missing or invalid arguments")
        exit(0)
    # create train and val dataset
    captcha_dataset = CaptchaDataset(config)
    train_dataset, val_dataset = captcha_dataset.create_dataset()
    # Train the model
    model_object = CTCModel(config)
    model = model_object.build_model(captcha_dataset.character_set)
    callback_list = model_object.callback()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=callback_list,
    )

    # model.save(config.model_save_dir)
    # char_to_num_file_path = config.model_save_dir + '/char_to_num.txt'
    # num_to_char_file_path = config.model_save_dir + '/num_to_char.txt'
    # with open(char_to_num_file_path, 'w') as f:
    #     for item in captcha_dataset.char_to_num.get_vocabulary():
    #         f.write("%s\n" % item)

    # with open(num_to_char_file_path, 'w') as f:
    #     for item in captcha_dataset.num_to_char.get_vocabulary():
    #         f.write("%s\n" % item)
    for batch in val_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds, captcha_dataset.max_length, captcha_dataset.num_to_char)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(captcha_dataset.num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
    plt.show()


if __name__ == '__main__':
    main()



