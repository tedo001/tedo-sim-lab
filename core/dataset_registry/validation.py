"""Checks a dataset's splits before a run trains on them: the same sample twice in one split, the
same sample in two splits (leakage: a test score measured on training data), one sample with two
labels, samples without a label, a class far rarer than the rest or missing from a split, and the
licence metadata of the card.

A runner describes its splits as :class:`SampleSet`\\ s (one content hash and label per sample,
exactly as it will split them); :func:`validate` turns them into a report. Problems that make a
score wrong (leakage, conflicting labels, missing labels) are *errors*; the rest are *warnings*.
The spec's ``data.validation`` rule says what errors do: ``warn`` records them, ``fail`` stops the
run before it trains, ``off`` skips the checks."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DataCheckFailed", "IMBALANCE_RATIO", "SampleSet", "digest", "licence_problems", "validate"]

#: A class this many times rarer than the most common one is reported.
IMBALANCE_RATIO = 10.0
_EXAMPLES = 5


class DataCheckFailed(ValueError):
    """The data checks found errors and the experiment's rule is ``fail``."""


def digest(data: bytes | str) -> str:
    """A short content hash of one sample (its bytes, or its text)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


@dataclass
class SampleSet:
    """One split as the run will use it."""
    hashes: Sequence[str]
    #: Label of each sample (class index or name), or ``None`` for unsupervised data.
    labels: Sequence[Any] | None = None
    #: Samples dropped or unusable because their label is missing.
    missing_labels: int = 0
    #: Class names by index, for messages.
    classes: Sequence[str] = field(default_factory=tuple)

    def name_of(self, label: Any) -> str:
        whole = isinstance(label, int | float) and float(label).is_integer()
        if whole and 0 <= int(label) < len(self.classes):
            return str(self.classes[int(label)])
        return str(label)


def _check(check_id: str, title: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"id": check_id, "title": title, "status": status, "detail": detail, **extra}


def _duplicates(name: str, samples: SampleSet) -> dict[str, Any]:
    counts = Counter(samples.hashes)
    repeated = {h: n for h, n in counts.items() if n > 1}
    extra = sum(n - 1 for n in repeated.values())
    if not extra:
        return _check(f"duplicates:{name}", f"Duplicates in {name}", "ok", "every sample is unique",
                      count=0)
    return _check(f"duplicates:{name}", f"Duplicates in {name}", "warn",
                  f"{extra:,} of {len(samples.hashes):,} samples repeat another ({len(repeated):,} distinct)",
                  count=extra, examples=sorted(repeated)[:_EXAMPLES])


def _leakage(first: str, a: SampleSet, second: str, b: SampleSet) -> dict[str, Any]:
    shared = set(a.hashes) & set(b.hashes)
    title = f"{first} ∩ {second}"
    if not shared:
        return _check(f"leakage:{first}:{second}", title, "ok", "no sample is in both", count=0)
    in_second = sum(1 for h in b.hashes if h in shared)
    return _check(f"leakage:{first}:{second}", title, "error",
                  f"{in_second:,} {second} samples ({in_second / max(len(b.hashes), 1):.1%}) also appear in "
                  f"{first}: their scores measure memory, not generalisation",
                  count=in_second, examples=sorted(shared)[:_EXAMPLES])


def _conflicts(splits: Mapping[str, SampleSet]) -> dict[str, Any]:
    labels: dict[str, set[str]] = defaultdict(set)
    for samples in splits.values():
        if samples.labels is None:
            continue
        for h, label in zip(samples.hashes, samples.labels, strict=False):
            labels[h].add(samples.name_of(label))
    clashing = {h: sorted(names) for h, names in labels.items() if len(names) > 1}
    if not clashing:
        return _check("label_conflicts", "Label conflicts", "ok", "no sample carries two labels", count=0)
    first = next(iter(clashing.items()))
    return _check("label_conflicts", "Label conflicts", "error",
                  f"{len(clashing):,} samples carry more than one label (e.g. {' / '.join(first[1])})",
                  count=len(clashing), examples=list(clashing)[:_EXAMPLES])


def _missing(splits: Mapping[str, SampleSet]) -> dict[str, Any]:
    missing = {name: s.missing_labels for name, s in splits.items() if s.missing_labels}
    if not missing:
        return _check("missing_labels", "Missing labels", "ok", "every sample has a label", count=0)
    total = sum(missing.values())
    where = ", ".join(f"{count:,} in {name}" for name, count in missing.items())
    return _check("missing_labels", "Missing labels", "error",
                  f"{total:,} samples have no label ({where}); they are left out of training and scoring",
                  count=total)


def _balance(splits: Mapping[str, SampleSet]) -> list[dict[str, Any]]:
    labelled = {name: s for name, s in splits.items() if s.labels is not None and len(s.labels)}
    if not labelled:
        return []
    out = []
    every = set().union(*(set(map(str, s.labels)) for s in labelled.values()))
    for name, samples in labelled.items():
        counts = Counter(map(str, samples.labels))
        absent = sorted(every - set(counts))
        largest, smallest = max(counts.values()), min(counts.values())
        ratio = largest / smallest
        rare = [samples.name_of(_number(label)) for label, n in counts.items()
                if largest / n >= IMBALANCE_RATIO]
        status, detail = "ok", f"{len(counts)} classes, largest/smallest {ratio:.1f}×"
        if absent:
            status = "warn"
            detail = (f"{len(absent)} class(es) never appear in {name}: "
                      f"{', '.join(samples.name_of(_number(a)) for a in absent[:_EXAMPLES])}")
        elif rare:
            status = "warn"
            detail = (f"largest/smallest class {ratio:.1f}×; rare: {', '.join(rare[:_EXAMPLES])} "
                      "(accuracy alone hides how these do; read per-class scores)")
        out.append(_check(f"balance:{name}", f"Class balance in {name}", status, detail,
                          counts={samples.name_of(_number(k)): v for k, v in sorted(counts.items())}))
    return out


def _number(text: str) -> Any:
    try:
        return int(text)
    except ValueError:
        return text


def licence_problems(card: Any) -> dict[str, Any]:
    """The card's licence metadata: a category, a named licence, a link to it, a citation."""
    if card is None:
        return _check("licence", "Licence metadata", "warn", "no dataset card: the licence is unknown")
    licence = card.license
    category = getattr(licence.category, "value", str(licence.category))
    gaps = []
    if category == "unspecified":
        gaps.append("no licence is published for it")
    if not licence.url:
        gaps.append("no link to the licence")
    if not getattr(card, "citation", None):
        gaps.append("no citation")
    if gaps:
        return _check("licence", "Licence metadata", "warn",
                      f"{licence.name} ({category}): " + "; ".join(gaps))
    return _check("licence", "Licence metadata", "ok", f"{licence.name} ({category}), cited, licence linked")


def validate(splits: Mapping[str, SampleSet], *, card: Any = None,
             extra: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """Every check over ``splits`` (split name → samples, in the order they are used: train first)."""
    checks = [_duplicates(name, samples) for name, samples in splits.items()]
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            checks.append(_leakage(first, splits[first], second, splits[second]))
    if any(s.labels is not None for s in splits.values()):
        checks += [_conflicts(splits), _missing(splits), *_balance(splits)]
    checks.append(licence_problems(card))
    checks += list(extra)
    errors = sum(1 for c in checks if c["status"] == "error")
    warnings = sum(1 for c in checks if c["status"] == "warn")
    return {"splits": {name: len(s.hashes) for name, s in splits.items()}, "checks": checks,
            "errors": errors, "warnings": warnings,
            "summary": "no problems" if not errors + warnings
            else f"{errors} error(s), {warnings} warning(s)"}
