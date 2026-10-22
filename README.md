# abc-gpt

A tiny (~0.8M parameter) character-level GPT trained on Irish traditional tunes in
[ABC notation](https://abcnotation.com/), using Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT). The training data is O'Neill's
*Music of Ireland* (1903) and *Dance Music of Ireland* (1907).

This repo only contains the project-specific files. `setup.sh` clones nanoGPT at a
pinned commit and symlinks these files into it.

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./setup.sh
git clone https://github.com/folkies/oneill ../oneill   # ABC transcriptions
```

## Train and sample

Run these from inside `nanoGPT/`:

```sh
python data/oneill/prepare.py --repo ../../oneill
python train.py config/train_oneill.py

python sample.py --out_dir=out-oneill --start=FILE:prompts/jig.txt \
    --num_samples=10 --max_new_tokens=500 --device=mps --dtype=float32 --compile=False \
    > samples.txt
python render.py samples.txt     # -> generated.abc
abc2midi generated.abc           # optional, -> generated1.mid, ...
```

The config targets Apple Silicon (`device = 'mps'`). Use `cuda` or `cpu` elsewhere.
To listen to the MIDI files, use `fluidsynth` with a General MIDI soundfont such as
[GeneralUser GS](https://schristiancollins.com/generaluser.php).

## How good is it?

Not bad for a model this size. The tunes are a bit repetitive, and some phrases
are clearly memorised from the books, since the corpus is small (~620K characters).
But the model does a good job of staying in key, and what comes out is music
rather than noise.

## Credits

The ABC transcriptions come from John Chambers' O'Neill's project, mirrored at
[folkies/oneill](https://github.com/folkies/oneill). They are © 1997–2000 their
transcribers and are used with credit. The music itself is public domain.
