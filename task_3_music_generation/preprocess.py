"""Convert a directory of MIDI files into note tokens for model training."""

import argparse
from pathlib import Path

import numpy as np
from music21 import chord, converter, instrument, note


def event_to_token(event) -> str | None:
    """Encode a music21 event as a compact, repeatable token."""
    if isinstance(event, note.Note):
        return str(event.pitch.midi)
    if isinstance(event, chord.Chord):
        return ".".join(str(pitch.midi) for pitch in event.pitches)
    if isinstance(event, note.Rest):
        return "rest"
    return None


def read_midi_file(path: Path) -> list[str]:
    """Parse one MIDI file and return its note, chord, and rest tokens."""
    score = converter.parse(path)
    parts = instrument.partitionByInstrument(score)
    event_stream = parts.parts[0].recurse() if parts else score.recurse()

    tokens = []
    for event in event_stream.notesAndRests:
        token = event_to_token(event)
        if token:
            tokens.append(token)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIDI note sequences for training.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    midi_files = sorted(args.input_dir.rglob("*.mid")) + sorted(args.input_dir.rglob("*.midi"))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {args.input_dir}")

    all_tokens: list[str] = []
    for midi_path in midi_files:
        try:
            tokens = read_midi_file(midi_path)
            all_tokens.extend(tokens)
            print(f"Read {midi_path.name}: {len(tokens)} tokens")
        except Exception as error:
            print(f"Skipped {midi_path.name}: {error}")

    if not all_tokens:
        raise RuntimeError("No usable musical events were found in the supplied files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tokens=np.array(all_tokens, dtype=str))
    print(f"Saved {len(all_tokens)} tokens to {args.output}")


if __name__ == "__main__":
    main()
