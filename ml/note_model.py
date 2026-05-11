"""
Note Denomination Classification Model.
Uses MobileNetV2 transfer learning for classifying Nepali currency notes.
Supports denominations: Rs.5, Rs.10, Rs.20, Rs.50, Rs.100, Rs.500, Rs.1000
This is a secondary advanced module of the Devanagari Character Recognition project.
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)


def build_note_model(num_classes: int = 7, input_shape=(128, 128, 3)):
    """
    Build a MobileNetV2-based transfer learning model for note classification.

    Architecture:
        MobileNetV2 (frozen base) -> GAP -> Dense(256) -> BN -> Dropout -> Dense(num_classes, softmax)
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base layers for transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last 20 layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_note_model_summary_text(model) -> str:
    """Return model.summary() as a string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
