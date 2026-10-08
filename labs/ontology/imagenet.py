"""The 1000 ImageNet-1k (ILSVRC 2012) classes in index order: WordNet id, names and gloss.

Read from ``data/imagenet1k_classes.tsv``, which holds the public class list only: the lab
never downloads or redistributes ImageNet images. Its sources and licences are in
``data/LICENSE-wordnet.txt`` (WordNet 3.0) and ``data/LICENSE-timm.txt`` (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

__all__ = ["DATA", "ImagenetClass", "imagenet_classes", "search_classes"]

DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class ImagenetClass:
    index: int
    #: WordNet noun id, "n" + 8-digit offset: "n01440764".
    wnid: str
    names: tuple[str, ...]
    gloss: str

    @property
    def name(self) -> str:
        return self.names[0]

    @property
    def offset(self) -> int:
        return int(self.wnid[1:])


@cache
def imagenet_classes() -> tuple[ImagenetClass, ...]:
    rows = []
    for line in (DATA / "imagenet1k_classes.tsv").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        index, wnid, names, gloss = line.split("\t")
        rows.append(ImagenetClass(int(index), wnid, tuple(n.strip() for n in names.split(",")), gloss))
    return tuple(rows)


def search_classes(text: str, limit: int = 200) -> list[ImagenetClass]:
    """Classes whose index, WordNet id, names or gloss contain ``text`` (case-insensitive);
    name matches first."""
    wanted = text.strip().lower()
    classes = imagenet_classes()
    if not wanted:
        return list(classes[:limit])
    if wanted.isdigit() and int(wanted) < len(classes):
        return [classes[int(wanted)]]
    named = [c for c in classes if wanted == c.wnid or any(wanted in n.lower() for n in c.names)]
    glossed = [c for c in classes if c not in named and wanted in c.gloss.lower()]
    return (named + glossed)[:limit]
