"""Train an LSTM next-note model from preprocessed MIDI tokens."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


SEQUENCE_LENGTH = 64


def make_training_data(tokens: list[str], vocabulary: list[str]):
    """Create normalised context sequences and next-token labels."""
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    encoded = np.array([token_to_id[token] for token in tokens], dtype=np.int32)

    if len(encoded) <= SEQUENCE_LENGTH:
        raise ValueError(f"At least {SEQUENCE_LENGTH + 1} tokens are required for training.")

    inputs = np.array(
        [encoded[index : index + SEQUENCE_LENGTH] for index in range(len(encoded) - SEQUENCE_LENGTH)],
        dtype=np.float32,
    )
    inputs /= len(vocabulary)
    targets = encoded[SEQUENCE_LENGTH:]
    return inputs[..., np.newaxis], targets


def build_model(vocabulary_size: int) -> tf.keras.Model:
    """Build a compact LSTM network for note prediction."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, 1)),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(vocabulary_size, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM music generator.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    tokens = np.load(args.dataset)["tokens"].tolist()
    vocabulary = sorted(set(tokens))
    inputs, targets = make_training_data(tokens, vocabulary)
    print(f"Training samples: {len(inputs)} | Vocabulary size: {len(vocabulary)}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    with (args.model_dir / "vocabulary.json").open("w", encoding="utf-8") as file:
        json.dump(vocabulary, file)

    model = build_model(len(vocabulary))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.model_dir / "best_music_lstm.keras", monitor="loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True),
    ]
    model.fit(inputs, targets, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    model.save(args.model_dir / "music_lstm.keras")
    print(f"Saved model and vocabulary in {args.model_dir}")


if __name__ == "__main__":
    main()
