"""
Prepare O'Neill's Music of Ireland (1903, "the 1850") and Dance Music of
Ireland (1907, "the 1001") for character-level training with nanoGPT.

Source: ABC transcriptions from John Chambers' O'Neill's project
(mirrored at https://github.com/folkies/oneill). The music is public domain;
the transcriptions are (c) 1997-2000 their contributors, who allow use with
credit to the transcribers. Credit them in anything you publish.

Usage (from the nanoGPT root):
    python data/oneill/prepare.py --repo path/to/oneill

Writes train.bin, val.bin, meta.pkl (for nanoGPT), train.txt / val.txt
(plain text, for browsing and memorisation checks) and tunes.jsonl
(per-tune metadata: book, number, title, type, transcriber, split).
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import re

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True, help="path to a clone of folkies/oneill")
parser.add_argument("--books", default="1850,1001", help="which books to include")
parser.add_argument("--val_frac", type=float, default=0.1, help="fraction of tune GROUPS held out")
parser.add_argument("--keep_phrasing", action="store_true",
                    help="keep slurs, staccato dots and the original spacing (off by default)")
parser.add_argument("--max_len", type=int, default=2000, help="drop tunes with longer bodies")
args = parser.parse_args()

out_dir = os.path.dirname(os.path.abspath(__file__))
ALLOWED = set(chr(c) for c in range(32, 127)) | {"\n"}

def read(path):
    raw = open(path, "rb").read().replace(b"\x00", b"").replace(b"\r", b"")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")

def header_field(header_lines, key):
    for ln in header_lines:
        if ln.startswith(key + ":"):
            return ln[2:].strip()
    return None

def default_unit_length(meter):
    # ABC standard: with no L: field, the unit is 1/16 if the meter's value
    # is below 0.75, otherwise 1/8.
    if meter in ("C", "C|", None):
        return "1/8"
    m = re.match(r"^(\d+)/(\d+)$", meter)
    if m and int(m.group(1)) / int(m.group(2)) < 0.75:
        return "1/16"
    return "1/8"

def clean_body(lines):
    out = []
    for ln in lines:
        ln = ln.split("%", 1)[0].rstrip()           # comments
        if not ln.strip():
            continue
        if re.match(r"^[A-Za-z]:", ln) and ln[0] not in "KMLP":
            continue                                # notes, lyrics, etc. inside the body
        ln = re.sub(r'"[^"]*~"N', "~", ln)          # 1001's custom roll macro -> standard roll
        ln = re.sub(r'"[^"]*"', "", ln)             # text annotations ("Slow", "D.C.", ...)
        if ln.strip():
            out.append(ln)
    return "\n".join(out)

def simplify(body):
    # Slurs and staccato dots are engraving detail from the printed books. A tiny
    # model can't keep brackets balanced, so they mostly add noise. Triplets
    # "(3" are rhythm, not phrasing, and are kept.
    body = re.sub(r"\((?!\d)", "", body)
    body = body.replace(")", "").replace(".", "")
    # Some transcriptions align bars in columns with runs of spaces; collapse
    # them so the model doesn't learn (and then run away with) long space runs.
    body = re.sub(r" {2,}", " ", body)
    return re.sub(r" *\n *", "\n", body).strip()

def title_group(title):
    # Crude normalisation so the same tune in both books (e.g. "The Girls of
    # Our Town" / "Girls of Our Town., The") lands on the same side of the split.
    t = (title or "").lower()
    t = re.sub(r"\bthe\b", "", t)
    return re.sub(r"[^a-z]", "", t)

def is_val(group):
    h = int(hashlib.md5(group.encode()).hexdigest(), 16)
    return (h % 10_000) < args.val_frac * 10_000

records, seen = [], set()
skipped = {"no_key": 0, "chars": 0, "length": 0, "duplicate": 0}

for book in args.books.split(","):
    for path in sorted(glob.glob(os.path.join(args.repo, book, "abc", "*.abc"))):
        text = read(path)
        lines = text.split("\n")
        k_idx = next((i for i, ln in enumerate(lines) if ln.startswith("K:")), None)
        if k_idx is None:
            skipped["no_key"] += 1
            continue
        header = lines[:k_idx]
        meter = header_field(header, "M")
        unit = header_field(header, "L") or default_unit_length(meter)
        key = lines[k_idx][2:].split("%")[0].strip()
        body = clean_body(lines[k_idx + 1:])
        if not args.keep_phrasing:
            body = simplify(body)

        if not set(body) <= ALLOWED or '"' in body:   # odd chars, or an unclosed annotation
            skipped["chars"] += 1
            continue
        if not body or len(body) > args.max_len:
            skipped["length"] += 1
            continue
        if body in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(body)

        title = header_field(header, "T")
        name = os.path.basename(path)
        records.append({
            "book": book,
            "number": header_field(header, "X"),
            "file": f"{book}/abc/{name}",
            "title": title,
            "type": header_field(header, "R"),
            "transcriber": name.split("_")[1][:-4] if "_" in name else None,
            "meter": meter,
            "key": key,
            "split": "val" if is_val(title_group(title)) else "train",
            # Only meter, unit length and key go into the training text: tune
            # type (R:) is missing for most of the 1850, so it can't be relied on.
            "abc": f"M:{meter or 'none'}\nL:{unit}\nK:{key}\n{body}\n\n",
        })

train_text = "".join(r["abc"] for r in records if r["split"] == "train")
val_text = "".join(r["abc"] for r in records if r["split"] == "val")

chars = sorted(set(train_text) | set(val_text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: np.array([stoi[c] for c in s], dtype=np.uint16)

encode(train_text).tofile(os.path.join(out_dir, "train.bin"))
encode(val_text).tofile(os.path.join(out_dir, "val.bin"))
with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(chars), "itos": itos, "stoi": stoi}, f)
open(os.path.join(out_dir, "train.txt"), "w").write(train_text)
open(os.path.join(out_dir, "val.txt"), "w").write(val_text)
with open(os.path.join(out_dir, "tunes.jsonl"), "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

n_train = sum(r["split"] == "train" for r in records)
print(f"kept {n_train} train / {len(records) - n_train} val tunes")
print(f"skipped: {skipped}")
print(f"train chars: {len(train_text):,}  val chars: {len(val_text):,}")
print(f"vocab size: {len(chars)}")
print("sample document:\n" + records[0]["abc"])
