from tensorflow.keras import Model, Sequential
import tensorflow as tf
from tensorflow import keras

def upsample(filters, size, strides, dropout=0.67):

    
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) 

    result = Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                               padding="same",
                                               kernel_initializer=initializer,
                                               use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if dropout:
        result.add(tf.keras.layers.Dropout(dropout))
    result.add(tf.keras.layers.LeakyReLU())
    return result

from tensorflow.keras.applications import VGG19

base_model = VGG19(input_shape=[256,256] + [3], include_top=False, weights="imagenet")

layers_names = [
    "block2_conv1",    # 256x256
    "block2_conv2",    # 256x256
    "block3_conv1",    # 128x128
    "block3_conv2",    # 128x128
    "block4_conv1",    # 64x64
    "block4_conv2",    # 64x64
    "block5_conv1",    # 32x32
]

layers = [base_model.get_layer(name).output for name in layers_names]
down_stack = Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False 

up_stack = [
    upsample(512, 3, 1),   # 32x32 -> 32x32
    upsample(512, 3, 2),   # 32x32 -> 64x64
    upsample(256, 3, 1),   # 64x64 -> 64x64
    upsample(256, 3, 2),   # 64x64 -> 128x128
    upsample(128, 3, 1),   # 128x128 -> 128x128
    upsample(128, 3, 2),   # 128x128 -> 256x256
]




keras.backend.clear_session()

def unet_generator(output_channels=1):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    x = inputs

    # Initializer for Conv2DTranspose layers
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Final output layer
    output = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, activation='sigmoid', 
        padding="same", kernel_initializer=initializer
    )

    concat = tf.keras.layers.Concatenate() 

    # Downsampling part
    skips = down_stack(x)
    x = skips[-1] 
    skips = reversed(skips[:-1]) 

    #Upsampling part with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)  
        if up.layers[0].strides == (2, 2):  
            x = concat([x, skip])  

            x = tf.keras.layers.Dropout(0.2)(x)


    # Final output layer
    x = output(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_generator()

