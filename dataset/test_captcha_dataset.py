import matplotlib.pyplot as plt
from dataset.captcha_dataset import CaptchaDataset
import tensorflow as tf
from utils import get_args, get_config_from_json

try:
    args = get_args()
    config, _ = get_config_from_json(args.config)
except Exception as e:
    print("missing or invalid arguments")
    exit(0)


captch_dataset = CaptchaDataset(config)
train_dataset, val_dataset = captch_dataset.create_dataset()

_, ax = plt.subplots(4, 4, figsize=(10, 5))

for batch in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(captch_dataset.num_to_char(labels[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.savefig('image.jpeg')
