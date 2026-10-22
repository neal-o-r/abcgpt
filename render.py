"""
Turn nanoGPT sample output into a playable .abc tunebook (and MIDI files).

Usage (from the nanoGPT root):
    python sample.py --out_dir=out-oneill --start=FILE:prompts/jig.txt \
        --num_samples=10 --max_new_tokens=500 --device=mps --dtype=float32 --compile=False \
        > samples.txt
    python render.py samples.txt            # writes generated.abc
    abc2midi generated.abc                  # optional: generated1.mid, generated2.mid, ...

Each sample is cut at its first blank line (the end-of-tune marker the model
learned from the training data), so trailing half-tunes are discarded.
"""
import re
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "samples.txt"
out = sys.argv[2] if len(sys.argv) > 2 else "generated.abc"

text = open(src, encoding="utf-8").read()
# sample.py separates samples with a line of dashes; drop any loader chatter.
raw_samples = [s for s in text.split("---------------") if "K:" in s]

tunes = []
for s in raw_samples:
    # skip any loader chatter before the first header line of the sample
    m = re.search(r"^[RM]:", s, re.M)
    s = s[m.start():] if m else s.lstrip()
    tune = s.split("\n\n")[0].strip()
    lines = tune.split("\n")
    # The model sometimes starts a new tune without a blank line first: cut
    # at the first header line that appears after the body has begun.
    for i, ln in enumerate(lines[3:], start=3):
        if re.match(r"^[MLK]:", ln):
            lines = lines[:i]
            break
    tune = "\n".join(lines)
    body = [ln for ln in lines if not (len(ln) > 1 and ln[1] == ":" and ln[0] in "RMLK")]
    if len(lines) < 5 or "|" not in "".join(body):
        continue  # header only, or no bar lines: not much of a tune
    tunes.append(tune)

with open(out, "w") as f:
    for i, tune in enumerate(tunes, 1):
        f.write(f"X:{i}\nT:Generated tune {i}\n{tune}\n\n")

print(f"{len(tunes)} of {len(raw_samples)} samples kept -> {out}")
