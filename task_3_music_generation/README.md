# Music Generation with AI

This project turns a collection of MIDI files into note sequences, trains an LSTM model on those sequences, and writes an original MIDI file from the trained model. The implementation uses `music21` for MIDI handling and TensorFlow/Keras for the neural network.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place legally obtained `.mid` or `.midi` files in `data/raw_midi/`. Small collections are useful for testing, but a varied collection of several hundred files produces more musical results.

## Run the pipeline

From this directory, run the commands in this order:

```powershell
python preprocess.py --input-dir data/raw_midi --output data/processed/notes.npz
python train.py --dataset data/processed/notes.npz --epochs 50 --model-dir models
python generate.py --model models/music_lstm.keras --vocabulary models/vocabulary.json --output output/generated_piece.mid
```

The generated file can be opened in a DAW, music notation application, or standard MIDI player.

## Implementation notes

- Notes, chords, and rests are encoded as tokens. A fixed-length context window is used to predict the next token.
- The LSTM learns the next-token probability distribution from the MIDI collection.
- Training time depends on the dataset and hardware. TensorFlow will use a supported GPU automatically when one is available.
- Generated music is intended as a starting point. Temperature sampling controls variety: lower values are safer; higher values are more experimental.
