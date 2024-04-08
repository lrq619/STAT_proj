import tensorflow as tf


def get_model(input_shape, num_classes=10):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Conv2D(72, kernel_size=1, use_bias=True))

    model.add(tf.keras.layers.Conv2D(48, kernel_size=3, use_bias=True))

    model.summary()

    return model


if __name__ == '__main__':
    input_shape = (32, 32, 3)
    get_model(input_shape)