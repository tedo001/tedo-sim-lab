"""Small stand-ins for downloaded datasets, written in each source's own layout: movie reviews (nltk
zip), SMS spam (UCI zip), 20 Newsgroups (by-date tarball) and spoken digits (WAV files)."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np

GOOD = ["a wonderful moving film with brilliant acting", "great story and a superb cast, loved it",
        "beautiful, funny and touching; the best film this year", "brilliant direction and great music"]
BAD = ["a dull boring mess and a waste of time", "terrible acting and an awful script",
       "boring, slow and badly made; avoid it", "the worst film, dull and stupid"]


def movie_reviews(root: Path, copies: int = 10) -> Path:
    path = root / "movie_reviews" / "movie_reviews.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for n in range(copies):
            for i, text in enumerate(GOOD):
                archive.writestr(f"movie_reviews/pos/cv{n}{i}_pos.txt", f"{text} ({n})\n")
            for i, text in enumerate(BAD):
                archive.writestr(f"movie_reviews/neg/cv{n}{i}_neg.txt", f"{text} ({n})\n")
    return path


def sms_spam(root: Path) -> Path:
    path = root / "sms_spam" / "sms_spam_collection.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ham\tsee you at lunch {i}" for i in range(20)] + \
            [f"spam\tWIN a FREE prize now, call {i} to claim cash" for i in range(20)]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("SMSSpamCollection", "\n".join(lines) + "\n")
        archive.writestr("readme", "the SMS Spam Collection")
    return path


def newsgroups(root: Path, groups: tuple[str, ...]) -> Path:
    path = root / "twenty_newsgroups" / "20news-bydate.tar.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for split in ("20news-bydate-train", "20news-bydate-test"):
            for g, group in enumerate(groups):
                for i in range(3 if split.endswith("train") else 2):
                    text = (f"From: someone@example.com\nSubject: post {i}\n\nAbout {group} number {g}.\n> "
                            "quoted text that should go\n\n-- \nsignature line\n")
                    data = text.encode("latin-1")
                    info = tarfile.TarInfo(f"{split}/{group}/{1000 + i}")
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
    return path


def spoken_digits(root: Path) -> Path:
    """Every FSDD file name, each a short tone whose pitch depends on the digit."""
    from labs.audio.datasets import file_names
    from labs.audio.dsp import save_wav

    folder = root / "spoken_digits" / "recordings"
    rate = 8000
    t = np.arange(int(0.3 * rate)) / rate
    tones = {d: (0.5 * np.sin(2 * np.pi * (300 + 150 * d) * t)) for d in range(10)}
    for name in file_names():
        save_wav(folder / name, tones[int(name[0])], rate)
    return folder
