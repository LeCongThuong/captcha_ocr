from utils import get_args, get_config_from_json
from models.ctc_model import CTCModel
from dataset.captcha_dataset import CaptchaDataset
import tensorflow as tf

try:
    args = get_args()
    config, _ = get_config_from_json(args.config)
except Exception as e:
    print(e)
    print("missing or invalid arguments")
    exit(0)

model_object = CTCModel(config)
captcha_dataset = CaptchaDataset(config)
characters_set = captcha_dataset.character_set
model = model_object.build_model(characters_set)
model.summary()

test_example = tf.zeros((1, 200, 50, 1))
label_example = tf.reshape(tf.constant([5, 6, 5, 5, 5]), shape=(1, 5))
example_dict = {"image": test_example, "label": label_example}
print(model.predict(example_dict))
