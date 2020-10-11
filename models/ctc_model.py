import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.ctc_loss_layer import CtcLayer


class CTCModel:
    def __init__(self, config):
        self.config = config

    def build_model(self, characters_set):
        input_img = layers.Input(shape=(self.config.img_width, self.config.img_height, 1), name='image',
                                 dtype=tf.float32)
        input_label = layers.Input(shape=(None, ), name='label', dtype=tf.float32)
        x = input_img
        for i in range(2):
            x = layers.Conv2D(self.config.num_kernels[i],
                              kernel_size=(3, 3),
                              activation="relu",
                              kernel_initializer="he_normal",
                              padding="same",
                              name=f"Conv{i+1}",
                              )(x)

            x = layers.MaxPooling2D((2, 2), name=f'pooling{i+1}')(x)

        new_shape = ((self.config.img_width // self.config.downsample_factor),
                     (self.config.img_height // self.config.downsample_factor) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(self.config.dropout)(x)

        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        x = layers.Dense(len(characters_set) + 1, activation="softmax", name="dense2")(x)
        print("Shape of input_label: ", input_label.shape)
        print("Shape of output: ", x.shape)
        # Add CTC layer for calculating CTC loss at each step
        output = CtcLayer(name="ctc_loss")(input_label, x)
        model = keras.models.Model(inputs=[input_img, input_label], outputs=output, name="ocr_model_v1")
        opt = keras.optimizers.Adam()
        # Compile the model and return
        model.compile(optimizer=opt)
        return model

    def callback(self):
        callback_list = []
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.config.early_stopping_patience, restore_best_weights=True
        )
        callback_list.append(early_stopping)

        return callback_list



