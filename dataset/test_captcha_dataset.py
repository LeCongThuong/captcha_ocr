import matplotlib.pyplot as plt
from dataset.captcha_dataset import CaptchaDataset
import tensorflow as tf


# captch_dataset = CaptchaDataset
# _, ax = plt.subplots(4, 4, figsize=(10, 5))
# for batch in train_dataset.take(1):
#     images = batch["image"]
#     labels = batch["label"]
#     for i in range(16):
#         img = (images[i] * 255).numpy().astype("uint8")
#         label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
#         ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
#         ax[i // 4, i % 4].set_title(label)
#         ax[i // 4, i % 4].axis("off")
# plt.show()
