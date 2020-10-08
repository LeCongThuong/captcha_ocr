from utils import get_args, get_config_from_json
from models.ctc_model import CTCModel
from dataset.captcha_dataset import CaptchaDataset

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
