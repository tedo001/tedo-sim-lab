"""WordNet 3.0 through nltk, for the Ontology Explorer.

The corpus is downloaded only when a person asks (Dataset Hub → WordNet → Download), into
``<workspace>/datasets/nltk_data`` so it travels with the project and never touches the
user's home folder. Behind a proxy, nltk refuses the download unless the person opts in with
``NLTK_ALLOW_PROXIED_URLOPEN=1``; its message says so.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from core.common.cancel import CancelToken, ProgressFn
from core.dataset_registry.adapter import DatasetAdapter, DatasetStats, LocalState

from .imagenet import ImagenetClass, imagenet_classes

__all__ = ["FOLDER", "Node", "SynsetInfo", "WordNet", "WordNetAdapter", "wnid_of"]

#: Under the workspace's datasets/ folder.
FOLDER = "nltk_data"


def _corpus(root: Path) -> Path:
    return root / FOLDER / "corpora"


def wnid_of(synset: Any) -> str:
    return f"{synset.pos()}{synset.offset():08d}"


class WordNetAdapter(DatasetAdapter):
    def data_dir(self, root: Path) -> Path:
        return root / FOLDER

    def local_status(self, root: Path) -> LocalState:
        corpus = _corpus(root)
        return "present" if (corpus / "wordnet.zip").is_file() or (corpus / "wordnet").is_dir() else "absent"

    def _download(self, root: Path, progress: ProgressFn, cancel: CancelToken) -> None:
        import nltk

        cancel.raise_if_cancelled()
        progress(-1.0, "Downloading WordNet with nltk…")
        nltk.download("wordnet", download_dir=str(self.data_dir(root)), quiet=True, raise_on_error=True)
        progress(1.0, "WordNet is ready")

    def describe(self, root: Path) -> DatasetStats:
        wordnet = WordNet(root)
        return DatasetStats({"noun synsets": wordnet.noun_count})


@dataclass(frozen=True)
class Node:
    """A synset as a link: its WordNet id, name ("dog.n.01") and first lemma."""

    wnid: str
    name: str
    lemma: str


@dataclass(frozen=True)
class SynsetInfo:
    node: Node
    lemmas: tuple[str, ...]
    definition: str
    examples: tuple[str, ...]
    #: Each path runs from the root ("entity") down to this synset.
    paths: tuple[tuple[Node, ...], ...]
    hyponyms: tuple[Node, ...]
    #: ImageNet-1k classes at or below this synset.
    imagenet: tuple[ImagenetClass, ...]


class WordNet:
    """A WordNet reader on the workspace's copy; raises ``LookupError`` when it is missing."""

    def __init__(self, root: Path) -> None:
        import nltk
        from nltk.corpus.reader.wordnet import WordNetCorpusReader

        folder = str(_corpus(root).parent)
        if folder not in nltk.data.path:  # nltk (3.10+) only reads corpora from its search path
            nltk.data.path.insert(0, folder)
        for resource in ("corpora/wordnet", "corpora/wordnet.zip/wordnet/"):
            try:
                location = nltk.data.find(resource, paths=[folder])
                break
            except LookupError:
                continue
        else:
            raise LookupError("WordNet is not downloaded: download it in the Dataset Hub")
        with warnings.catch_warnings():  # no Open Multilingual Wordnet: English only, as intended
            warnings.simplefilter("ignore", UserWarning)
            self.reader = WordNetCorpusReader(location, None)

    @property
    def noun_count(self) -> int:
        return sum(1 for _ in self.reader.all_synsets("n"))

    @staticmethod
    def node(synset: Any) -> Node:
        return Node(wnid_of(synset), synset.name(), synset.lemmas()[0].name().replace("_", " "))

    def synset(self, key: str) -> Any:
        """By WordNet id ("n02084071") or name ("dog.n.01"); ``LookupError`` if unknown."""
        try:
            with warnings.catch_warnings():  # nltk warns (and returns None) for unknown offsets
                warnings.simplefilter("ignore", UserWarning)
                if len(key) == 9 and key[0] in "nvar" and key[1:].isdigit():
                    found = self.reader.synset_from_pos_and_offset(key[0], int(key[1:]))
                else:
                    found = self.reader.synset(key)
        except Exception as exc:  # nltk raises WordNetError, ValueError or IndexError
            raise LookupError(f"no WordNet synset {key!r}") from exc
        if found is None:
            raise LookupError(f"no WordNet synset {key!r}")
        return found

    def lookup(self, word: str, limit: int = 30) -> list[Node]:
        """Noun synsets for ``word`` ("dog", "hot dog")."""
        word = word.strip().replace(" ", "_")
        return [self.node(s) for s in self.reader.synsets(word, pos="n")[:limit]] if word else []

    @cached_property
    def _ancestors(self) -> dict[str, list[ImagenetClass]]:
        """WordNet id → the ImageNet classes at or below it."""
        below: dict[str, list[ImagenetClass]] = {}
        for item in imagenet_classes():
            synset = self.synset(item.wnid)
            seen = {wnid_of(s) for path in synset.hypernym_paths() for s in path}
            for wnid in seen:
                below.setdefault(wnid, []).append(item)
        return below

    def info(self, key: str) -> SynsetInfo:
        synset = self.synset(key)
        paths = tuple(tuple(self.node(s) for s in path) for path in synset.hypernym_paths())
        hyponyms = sorted(synset.hyponyms() + synset.instance_hyponyms(), key=lambda s: s.name())
        lemmas = tuple(lemma.name().replace("_", " ") for lemma in synset.lemmas())
        return SynsetInfo(self.node(synset), lemmas,
                          synset.definition(), tuple(synset.examples()), paths,
                          tuple(self.node(s) for s in hyponyms),
                          tuple(self._ancestors.get(wnid_of(synset), ())))
