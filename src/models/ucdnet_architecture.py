"""UCDNet Keras architecture (encoder + NSPP + decoder)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.losses import combined_loss, make_loss


class BilinearResize(layers.Layer):
    """Resize tensor x to match reference spatial dimensions."""

    def call(self, inputs):
        x, reference = inputs
        target = tf.shape(reference)
        return tf.image.resize(x, [target[1], target[2]], method="bilinear")


def pooling_block(f_enc, out_channels: int, stride: int):
    b1 = layers.DepthwiseConv2D(3, strides=stride, padding="same")(f_enc)
    b1 = layers.Conv2D(out_channels, 1, padding="same")(b1)
    b2 = layers.AveragePooling2D(pool_size=stride, strides=stride, padding="same")(f_enc)
    b2 = layers.Conv2D(out_channels, 1, padding="same")(b2)
    fp = layers.Add()([b1, b2])
    return layers.ReLU()(fp)


def global_pooling_block(f_p, f_enc_ref, out_channels: int):
    fg = layers.GlobalAveragePooling2D(keepdims=True)(f_p)
    fg = layers.Conv2D(out_channels, 1, padding="same")(fg)
    fp_up = BilinearResize()([fg, f_enc_ref])
    fp_up = layers.Conv2D(out_channels, 3, padding="same")(fp_up)
    return layers.ReLU()(fp_up)


def nspp_block(f_enc):
    channels = f_enc.shape[-1]
    out_c = max(1, channels // 4)
    paths = []
    for stride in (2, 4, 8, 16):
        fp = pooling_block(f_enc, out_c, stride)
        paths.append(global_pooling_block(fp, f_enc, out_c))
    concat = layers.Concatenate()(paths + [f_enc])
    out = layers.Conv2D(channels, 1, padding="same")(concat)
    return layers.ReLU()(out)


def encoder_stage(x1, x2, filters: int, n_convs: int, name: str):
    conv1 = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c1")
    c1_1 = layers.ReLU()(conv1(x1))
    c1_2 = layers.ReLU()(conv1(x2))

    diff = layers.Subtract(name=f"{name}_diff")([c1_1, c1_2])
    cr = layers.Conv2D(filters, 1, padding="same", name=f"{name}_res")(diff)
    cr = layers.ReLU()(cr)

    out1, out2 = c1_1, c1_2
    for j in range(2, n_convs + 1):
        conv_j = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c{j}")
        out1 = layers.ReLU()(conv_j(out1))
        out2 = layers.ReLU()(conv_j(out2))

    out1 = layers.Concatenate(name=f"{name}_cat1")([out1, cr])
    out2 = layers.Concatenate(name=f"{name}_cat2")([out2, cr])
    return out1, out2


def decoder_stage(x, skip1, skip2, filters: int, n_convs: int, name: str):
    x = layers.UpSampling2D(2, name=f"{name}_up")(x)
    x = layers.Conv2D(filters, 2, padding="same", name=f"{name}_upconv")(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate(name=f"{name}_cat")([skip1, x, skip2])
    for j in range(n_convs):
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv{j + 1}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{j + 1}")(x)
        x = layers.ReLU()(x)
    return x


def build_ucdnet(
    patch_size: int | None = 512,
    num_bands: int = 13,
    num_classes: int = 2,
) -> keras.Model:
    """
    Build UCDNet.

    patch_size=None yields a fully convolutional model for arbitrary image sizes
    at inference time.
    """
    spatial = patch_size if patch_size is not None else None
    inp_t1 = keras.Input((spatial, spatial, num_bands), name="T1")
    inp_t2 = keras.Input((spatial, spatial, num_bands), name="T2")

    skips_1, skips_2 = [], []

    o1, o2 = encoder_stage(inp_t1, inp_t2, 16, 1, "s1")
    skips_1.append(o1)
    skips_2.append(o2)
    o1 = layers.MaxPooling2D(2, name="pool1_1")(o1)
    o2 = layers.MaxPooling2D(2, name="pool1_2")(o2)

    o1, o2 = encoder_stage(o1, o2, 32, 2, "s2")
    skips_1.append(o1)
    skips_2.append(o2)
    o1 = layers.MaxPooling2D(2, name="pool2_1")(o1)
    o2 = layers.MaxPooling2D(2, name="pool2_2")(o2)

    o1, o2 = encoder_stage(o1, o2, 64, 3, "s3")
    skips_1.append(o1)
    skips_2.append(o2)
    o1 = layers.MaxPooling2D(2, name="pool3_1")(o1)
    o2 = layers.MaxPooling2D(2, name="pool3_2")(o2)

    o1, o2 = encoder_stage(o1, o2, 128, 3, "s4")
    extra = layers.Conv2D(64, 3, padding="same", name="s4_extra")
    o1 = layers.ReLU()(extra(o1))
    o2 = layers.ReLU()(extra(o2))

    diff = layers.Subtract(name="enc_diff")([o1, o2])
    f_enc = layers.Concatenate(name="enc_out")([o1, diff, o2])

    f_nspp = nspp_block(f_enc)
    x = decoder_stage(f_nspp, skips_1[2], skips_2[2], 64, 3, "dec1")
    x = decoder_stage(x, skips_1[1], skips_2[1], 32, 2, "dec2")
    x = decoder_stage(x, skips_1[0], skips_2[0], 16, 1, "dec3")

    x = layers.Conv2D(num_classes, 1, padding="same", name="final_conv")(x)
    output = layers.Softmax(name="change_map")(x)

    return keras.Model(inputs=[inp_t1, inp_t2], outputs=output, name="UCDNet")


def get_custom_objects() -> dict:
    return {
        "BilinearResize": BilinearResize,
        "combined_loss": combined_loss,
        "ucdnet_combined_loss": make_loss(),
    }
