"""Text-classification datasets. Each is read straight from the archive its authors (or nltk)
publish — no pickles, no client libraries — into ``<workspace>/datasets/<card id>/``.

- Movie Review Data (polarity 2.0, Pang & Lee 2004): 1,000 positive and 1,000 negative reviews,
  from nltk's data repository.
- SMS Spam Collection (Almeida & Hidalgo 2011, CC BY 4.0): 5,574 messages, ham or spam, from UCI.
- 20 Newsgroups (by-date version, Lang 1995): ~18,800 posts in 20 groups with the authors' own
  train/test split by date; headers, signatures and quotes are removed so a model must read the
  text (as scikit-learn's ``remove=("headers", "footers", "quotes")``)."""

from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path
from typing import ClassVar, Literal

from core.common.cancel import CancelToken, ProgressFn
from core.dataset_registry.adapter import DatasetStats, LocalState, TextAdapter

from ..common.download import download_file

__all__ = ["MovieReviewsAdapter", "NewsgroupsAdapter", "SmsSpamAdapter", "clean_post"]

Split = Literal["train", "test", "all"]


class _ArchiveText(TextAdapter):
    url: ClassVar[str]
    archive: ClassVar[str]
    md5: ClassVar[str | None] = None

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.card.class_names)

    def local_status(self, root: Path) -> LocalState:
        path = self.data_dir(root) / self.archive
        return "present" if path.is_file() else "absent"

    def _download(self, root: Path, progress: ProgressFn, cancel: CancelToken) -> None:
        download_file([self.url], self.data_dir(root) / self.archive, md5=self.md5, progress=progress,
                      cancel=cancel, timeout=120)
        progress(1.0, f"{self.card.name} is ready")

    def describe(self, root: Path) -> DatasetStats:
        return DatasetStats(dict(self.card.splits), len(self.classes), self.classes, {"kind": "text"})

    def _need(self, root: Path) -> Path:
        path = self.data_dir(root) / self.archive
        if not path.is_file():
            raise FileNotFoundError(f"{self.card.name} is not downloaded. Download it from the Dataset Hub, "
                                    "the NLP / LLM lab or the Experiment Builder first.")
        return path


class MovieReviewsAdapter(_ArchiveText):
    url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip"
    archive = "movie_reviews.zip"

    def load_texts(self, root: Path, split: Split) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        with zipfile.ZipFile(self._need(root)) as archive:
            for name in sorted(archive.namelist()):
                match = re.search(r"/(neg|pos)/[^/]+\.txt$", name)
                if match:
                    texts.append(archive.read(name).decode("utf-8", errors="replace"))
                    labels.append(self.classes.index(match.group(1)))
        return texts, labels


class SmsSpamAdapter(_ArchiveText):
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    archive = "sms_spam_collection.zip"

    def load_texts(self, root: Path, split: Split) -> tuple[list[str], list[int]]:
        texts, labels = [], []
        with zipfile.ZipFile(self._need(root)) as archive:
            name = next(n for n in archive.namelist() if n.rsplit("/", 1)[-1] == "SMSSpamCollection")
            for line in archive.read(name).decode("utf-8", errors="replace").splitlines():
                kind, _, text = line.partition("\t")
                if kind in self.classes and text:
                    texts.append(text)
                    labels.append(self.classes.index(kind))
        return texts, labels


_QUOTE = re.compile(r"(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)")


def clean_post(text: str) -> str:
    """A newsgroup post without its header block, quoted lines and signature."""
    _header, _, body = text.partition("\n\n")
    lines = [line for line in body.splitlines() if not _QUOTE.search(line)]
    body = "\n".join(lines).strip()
    cut = body.rfind("\n-- \n")  # the signature marker
    if cut == -1:
        cut = body.rfind("\n--\n")
    return body[:cut] if cut > 0 else body


class NewsgroupsAdapter(_ArchiveText):
    url = "https://ndownloader.figshare.com/files/5975967"
    archive = "20news-bydate.tar.gz"

    def load_texts(self, root: Path, split: Split) -> tuple[list[str], list[int]]:
        wanted = {"train": ("20news-bydate-train",), "test": ("20news-bydate-test",),
                  "all": ("20news-bydate-train", "20news-bydate-test")}[split]
        texts, labels = [], []
        with tarfile.open(self._need(root), "r:gz") as archive:
            for member in archive:
                parts = member.name.split("/")
                if (not member.isfile() or len(parts) < 3 or parts[-3] not in wanted
                        or parts[-2] not in self.classes):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                texts.append(clean_post(handle.read().decode("latin-1")))
                labels.append(self.classes.index(parts[-2]))
        return texts, labels
