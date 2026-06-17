"""
ucdnet_architecture.py — UCDNet Model Architecture

Network layout:
  Encoder (4 stages, shared-weight siamese branches)
    └─ Stage 1 : 2× Conv3×3 → 16 filters  + difference branch
    └─ Stage 2 : 2× Conv3×3 → 32 filters  + difference branch
    └─ Stage 3 : 3× Conv3×3 → 64 filters  + difference branch
    └─ Stage 4 : 3× Conv3×3 → 128 filters + difference branch → bottleneck
  NSPP Block  (Nested Spatial Pyramid Pooling, Eq. 4–7)
  Decoder (3 stages with skip connections from both branches)
    └─ Stage 1 : upsample + concat(skip_s3_t1, skip_s3_t2) → 32 ch
    └─ Stage 2 : upsample + concat(skip_s2_t1, skip_s2_t2) → 16 ch
    └─ Stage 3 : upsample + concat(skip_s1_t1, skip_s1_t2) → 16 ch
  Output : Conv1×1 → softmax  (2 classes: unchanged / changed)

Usage
-----
    from src.models.ucdnet_architecture import build_ucdnet
    model = build_ucdnet(input_shape=(512, 512, 13), num_classes=2)
    model.summary()
"""

import tensorflow as tf
from tensorflow.keras import Model, layers

# ── Low-level conv helpers ────────────────────────────────────────────────


def _conv3x3(x, filters, name):
    x = layers.Conv2D(filters, kernel_size=3, padding="same", name=name)(x)
    x = layers.Activation("relu")(x)
    return x


def _conv1x1(x, filters, name):
    x = layers.Conv2D(filters, kernel_size=1, padding="same", name=name)(x)
    x = layers.Activation("relu")(x)
    return x


def _conv2x2(x, filters, name):
    x = layers.Conv2D(filters, kernel_size=2, padding="same", name=name)(x)
    x = layers.Activation("relu")(x)
    return x


#  Encoder stages


def _encoder_stage1(t1, t2):
    """
    Stage 1 — shared 16-filter siamese branch.
    Returns skip connections (for decoder) and downsampled outputs.
    """
    shared_conv1 = layers.Conv2D(
        16, 3, padding="same", activation="relu", name="s1_conv1"
    )
    shared_conv2 = layers.Conv2D(
        16, 3, padding="same", activation="relu", name="s1_conv2"
    )

    f1 = shared_conv1(t1)
    f2 = shared_conv1(t2)
    diff = layers.Subtract(name="s1_diff")([f1, f2])
    cr = _conv1x1(diff, 16, name="s1_cr")

    f1 = shared_conv2(f1)
    f2 = shared_conv2(f2)

    skip1 = layers.Concatenate(name="s1_skip1")([f1, cr])
    skip2 = layers.Concatenate(name="s1_skip2")([f2, cr])

    pool = layers.MaxPool2D(2, name="s1_pool")
    return skip1, skip2, pool(skip1), pool(skip2)


def _encoder_stage2(x1, x2):
    """Stage 2 — shared 32-filter siamese branch."""
    shared_conv1 = layers.Conv2D(
        32, 3, padding="same", activation="relu", name="s2_conv1"
    )
    shared_conv2 = layers.Conv2D(
        32, 3, padding="same", activation="relu", name="s2_conv2"
    )

    f1 = shared_conv1(x1)
    f2 = shared_conv1(x2)
    diff = layers.Subtract(name="s2_diff")([f1, f2])
    cr = _conv1x1(diff, 32, name="s2_cr")

    f1 = shared_conv2(f1)
    f2 = shared_conv2(f2)

    skip1 = layers.Concatenate(name="s2_skip1")([f1, cr])
    skip2 = layers.Concatenate(name="s2_skip2")([f2, cr])

    pool = layers.MaxPool2D(2, name="s2_pool")
    return skip1, skip2, pool(skip1), pool(skip2)


def _encoder_stage3(x1, x2):
    """Stage 3 — shared 64-filter siamese branch (3 convs per branch)."""
    shared_conv1 = layers.Conv2D(
        64, 3, padding="same", activation="relu", name="s3_conv1"
    )
    shared_conv2 = layers.Conv2D(
        64, 3, padding="same", activation="relu", name="s3_conv2"
    )
    shared_conv3 = layers.Conv2D(
        64, 3, padding="same", activation="relu", name="s3_conv3"
    )

    f1 = shared_conv1(x1)
    f2 = shared_conv1(x2)
    diff = layers.Subtract(name="s3_diff")([f1, f2])
    cr = _conv1x1(diff, 64, name="s3_cr")

    f1 = shared_conv3(shared_conv2(f1))
    f2 = shared_conv3(shared_conv2(f2))

    skip1 = layers.Concatenate(name="s3_skip1")([f1, cr])
    skip2 = layers.Concatenate(name="s3_skip2")([f2, cr])

    pool = layers.MaxPool2D(2, name="s3_pool")
    return skip1, skip2, pool(skip1), pool(skip2)


def _encoder_stage4(x1, x2):
    """
    Stage 4 — shared 128-filter siamese branch.
    Outputs a single fused bottleneck feature map (no separate skips needed).
    """
    shared_conv1 = layers.Conv2D(
        128, 3, padding="same", activation="relu", name="s4_conv1"
    )
    shared_conv2 = layers.Conv2D(
        128, 3, padding="same", activation="relu", name="s4_conv2"
    )
    shared_conv3 = layers.Conv2D(
        128, 3, padding="same", activation="relu", name="s4_conv3"
    )

    f1 = shared_conv1(x1)
    f2 = shared_conv1(x2)
    diff = layers.Subtract(name="s4_diff")([f1, f2])
    cr = _conv1x1(diff, 128, name="s4_cr")

    f1 = shared_conv3(shared_conv2(f1))
    f2 = shared_conv3(shared_conv2(f2))

    feat1 = layers.Concatenate(name="s4_feat1")([f1, cr])
    feat2 = layers.Concatenate(name="s4_feat2")([f2, cr])

    r1 = _conv1x1(feat1, 64, name="s4_reduce1")
    r2 = _conv1x1(feat2, 64, name="s4_reduce2")
    diff_r = layers.Subtract(name="s4_diff_final")([r1, r2])

    return layers.Concatenate(name="s4_enc_out")([r1, r2, diff_r])  # (B,H,W,192)


#  NSPP Block


def _nspp_block(f_enc):
    """
    Nested Spatial Pyramid Pooling block (Eq. 4–7).

    Input  : f_enc   → (B, H, W, C)
    Output : fnew    → (B, H, W, C)   same spatial size and channel count

    Four parallel paths with strides {2, 4, 8, 16}:
      Eq. 4 — strided SeparableConv + AveragePool branch, fused with Add
      Eq. 5 — GlobalAveragePool per path  → (B, 1, 1, C/4)
      Eq. 6 — Upsample back to (H, W) via UpSampling2D + Conv3×3
      Eq. 7 — Concatenate all 4 paths + f_enc, project back to C channels
    """
    C = f_enc.shape[-1]  # 192 for default config
    c4 = C // 4  # channels per path
    H_static = f_enc.shape[1]
    W_static = f_enc.shape[2]

    path_outputs = []

    for stride in [2, 4, 8, 16]:

        # Eq. 4 — strided branch
        b_conv = layers.SeparableConv2D(
            c4, 3, strides=stride, padding="same", name=f"nspp_strided_s{stride}"
        )(f_enc)

        b_avg = layers.AveragePooling2D(
            pool_size=stride, strides=stride, padding="same", name=f"nspp_avg_s{stride}"
        )(f_enc)
        b_avg = layers.Conv2D(c4, 1, padding="same", name=f"nspp_avg_pw_s{stride}")(
            b_avg
        )

        fp = layers.Activation("relu", name=f"nspp_relu_s{stride}")(
            layers.Add(name=f"nspp_add_s{stride}")([b_conv, b_avg])
        )

        # Eq. 5 — global context per path
        fg = layers.Reshape((1, 1, c4), name=f"nspp_reshape_s{stride}")(
            layers.GlobalAveragePooling2D(name=f"nspp_gap_s{stride}")(fp)
        )
        fg = layers.Conv2D(c4, 1, padding="same", name=f"nspp_gpw_s{stride}")(fg)

        # Eq. 6 — upsample back to original spatial size
        if H_static is not None and W_static is not None:
            fp_out = layers.UpSampling2D(
                size=(H_static, W_static),
                interpolation="bilinear",
                name=f"nspp_up_s{stride}",
            )(fg)
        else:
            # Dynamic shape fallback (e.g. when H/W are unknown at build time)
            fp_out = layers.Lambda(
                lambda inputs: tf.image.resize(
                    inputs[0],
                    (tf.shape(inputs[1])[1], tf.shape(inputs[1])[2]),
                    method="bilinear",
                ),
                name=f"nspp_up_dyn_s{stride}",
            )([fg, f_enc])

        fp_out = layers.Conv2D(
            c4,
            3,
            padding="same",
            activation="relu",
            name=f"nspp_post_up_conv_s{stride}",
        )(fp_out)

        path_outputs.append(fp_out)

    # Eq. 7 — aggregate all paths + skip
    cat = layers.Concatenate(name="nspp_cat")(path_outputs + [f_enc])
    return layers.Conv2D(C, 1, padding="same", activation="relu", name="nspp_out")(cat)


#  Decoder stages


def _decoder_stage1(x, skip1_s3, skip2_s3):
    """Decoder stage 1 — upsample × 2, concat stage-3 skips → 32 ch."""
    x = _conv2x2(layers.UpSampling2D(2, name="d1_up")(x), 64, name="d1_up_conv")
    x = layers.Concatenate(name="d1_cat")([x, skip1_s3, skip2_s3])
    x = _conv3x3(x, 64, name="d1_conv1")
    x = _conv3x3(x, 64, name="d1_conv2")
    x = _conv3x3(x, 32, name="d1_conv3")
    x = layers.Dropout(0.3, name="d1_drop")(x)
    x = layers.BatchNormalization(name="d1_bn")(x)
    return x


def _decoder_stage2(x, skip1_s2, skip2_s2):
    """Decoder stage 2 — upsample × 2, concat stage-2 skips → 16 ch."""
    x = _conv2x2(layers.UpSampling2D(2, name="d2_up")(x), 32, name="d2_up_conv")
    x = layers.Concatenate(name="d2_cat")([x, skip1_s2, skip2_s2])
    x = _conv3x3(x, 32, name="d2_conv1")
    x = _conv3x3(x, 16, name="d2_conv2")
    x = layers.Dropout(0.3, name="d2_drop")(x)
    x = layers.BatchNormalization(name="d2_bn")(x)
    return x


def _decoder_stage3(x, skip1_s1, skip2_s1):
    """Decoder stage 3 — upsample × 2, concat stage-1 skips → 16 ch."""
    x = _conv2x2(layers.UpSampling2D(2, name="d3_up")(x), 16, name="d3_up_conv")
    x = layers.Concatenate(name="d3_cat")([x, skip1_s1, skip2_s1])
    x = _conv3x3(x, 16, name="d3_conv1")
    x = layers.BatchNormalization(name="d3_bn")(x)
    return x


#  Public builder


def build_ucdnet(input_shape=(512, 512, 13), num_classes=2):
    """
    Build and return the UCDNet Keras model.

    Parameters
    ----------
    input_shape : tuple  (H, W, bands)  — default (512, 512, 13)
    num_classes : int                   — default 2  (unchanged / changed)

    Returns
    -------
    tf.keras.Model  with named inputs "T1" and "T2"
    """
    T1 = layers.Input(shape=input_shape, name="T1")
    T2 = layers.Input(shape=input_shape, name="T2")

    # Encoder
    skip1_s1, skip2_s1, o1, o2 = _encoder_stage1(T1, T2)
    skip1_s2, skip2_s2, o1, o2 = _encoder_stage2(o1, o2)
    skip1_s3, skip2_s3, o1, o2 = _encoder_stage3(o1, o2)
    f_enc = _encoder_stage4(o1, o2)

    # NSPP bottleneck
    f_nspp = _nspp_block(f_enc)

    # Decoder
    d1 = _decoder_stage1(f_nspp, skip1_s3, skip2_s3)
    d2 = _decoder_stage2(d1, skip1_s2, skip2_s2)
    d3 = _decoder_stage3(d2, skip1_s1, skip2_s1)

    # Output head
    output = layers.Conv2D(
        num_classes, 1, padding="same", activation="softmax", name="change_map"
    )(d3)

    return Model(inputs=[T1, T2], outputs=output, name="UCDNet")


#  Quick smoke-test

if __name__ == "__main__":
    import numpy as np

    print("Building UCDNet ...")
    model = build_ucdnet(input_shape=(512, 512, 13), num_classes=2)
    model.summary()

    print("\nRunning forward pass with dummy data ...")
    dummy = np.random.rand(1, 512, 512, 13).astype("float32")
    out = model.predict({"T1": dummy, "T2": dummy}, verbose=0)

    assert out.shape == (1, 512, 512, 2), f"Unexpected output shape: {out.shape}"
    print(f"Output shape : {out.shape}  ✓")
    print(f"Total params : {model.count_params():,}")
