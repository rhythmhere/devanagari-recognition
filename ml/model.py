"""
Advanced CNN Model for Devanagari Character Recognition.
Architecture: 4 convolutional blocks with BatchNorm + Dropout,
Global Average Pooling, Dense head with L2 regularisation.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2


def build_model(num_classes: int, input_shape=(32, 32, 1)):
    """
    Build an advanced sequential CNN for Devanagari character classification.

    Architecture overview:
        Block 1: Conv(32) -> BN -> ReLU -> Conv(32) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
        Block 2: Conv(64) -> BN -> ReLU -> Conv(64) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
        Block 3: Conv(128) -> BN -> ReLU -> MaxPool -> Dropout(0.3)
        Block 4: Conv(256) -> BN -> ReLU -> GlobalAveragePooling
        Head:    Dense(512) -> BN -> Dropout(0.5) -> Dense(num_classes, softmax)
    """
    model = Sequential([
        # ── Block 1 ────────────────────────────────────
        Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4), input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # ── Block 2 ────────────────────────────────────
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # ── Block 3 ────────────────────────────────────
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # ── Block 4 ────────────────────────────────────
        Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        # ── Classification Head ────────────────────────
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_model_summary_text(model) -> str:
    """Return model.summary() as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
