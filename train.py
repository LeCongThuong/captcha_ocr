import tensorflow
from tensorflow import keras
from utils import get_args, get_config_from_json
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

    model.save(config.model_save_dir)
    char_to_num_file_path = config.model_save_dir + '/char_to_num.txt'
    num_to_char_file_path = config.model_save_dir + '/num_to_char.txt'
    with open(char_to_num_file_path, 'w') as f:
        for item in captcha_dataset.char_to_num.get_vocabulary():
            f.write("%s\n" % item)

    with open(num_to_char_file_path, 'w') as f:
        for item in captcha_dataset.num_to_char.get_vocabulary():
            f.write("%s\n" % item)


if __name__ == '__main__':
    main()



