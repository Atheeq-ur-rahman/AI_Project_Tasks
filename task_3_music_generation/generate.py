"""Generate a MIDI sequence from a trained LSTM music model."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from music21 import chord, note, stream


SEQUENCE_LENGTH = 64


def sample_index(probabilities: np.ndarray, temperature: float) -> int:
    """Sample a token index after applying temperature to the model output."""
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-8) / temperature
    probabilities = np.exp(probabilities - probabilities.max())
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


def token_to_event(token: str):
    """Turn a saved token back into a music21 event."""
    if token == "rest":
        return note.Rest(quarterLength=0.5)
    if "." in token:
        return chord.Chord([int(value) for value in token.split(".")], quarterLength=0.5)
    return note.Note(int(token), quarterLength=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a MIDI file from a trained model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(args.model)
    seed = np.random.randint(0, len(vocabulary), size=SEQUENCE_LENGTH).tolist()
    generated_tokens: list[str] = []

    for _ in range(args.length):
        sequence = np.array(seed[-SEQUENCE_LENGTH:], dtype=np.float32) / len(vocabulary)
        prediction = model.predict(sequence.reshape(1, SEQUENCE_LENGTH, 1), verbose=0)[0]
        next_id = sample_index(prediction, args.temperature)
        generated_tokens.append(vocabulary[next_id])
        seed.append(next_id)

    music_stream = stream.Stream(token_to_event(token) for token in generated_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    music_stream.write("midi", fp=args.output)
    print(f"Generated {len(generated_tokens)} events in {args.output}")


if __name__ == "__main__":
    main()
