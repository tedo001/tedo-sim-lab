"""Syntax colouring for the code editor: Python (with multi-line strings), YAML, JSON,
Markdown and TOML/INI. Rules are regular expressions; colours come from ``tokens.py``."""

from __future__ import annotations

import builtins
import keyword
import re
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextDocument

from ..theme.tokens import COLORS

__all__ = ["highlighter_for"]


def _style(colour: str, *, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    style = QTextCharFormat()
    style.setForeground(QColor(COLORS[colour]))
    if bold:
        style.setFontWeight(QFont.Weight.DemiBold)
    style.setFontItalic(italic)
    return style


class _RuleHighlighter(QSyntaxHighlighter):
    #: (pattern, style key, capture group) applied in order; later rules win.
    RULES: tuple[tuple[str, str, int], ...] = ()

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)
        self._rules = [(re.compile(pattern), _style(colour, bold=colour == "code_keyword",
                                                    italic=colour == "code_comment"), group)
                       for pattern, colour, group in self.RULES]

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt API)
        for pattern, style, group in self._rules:
            for match in pattern.finditer(text):
                start, end = match.span(group)
                if end > start:
                    self.setFormat(start, end - start, style)


_PY_KEYWORDS = "|".join(sorted(keyword.kwlist + list(getattr(keyword, "softkwlist", [])), key=len,
                               reverse=True))
_PY_BUILTINS = "|".join(name for name in dir(builtins) if not name.startswith("_") and name.isidentifier())


class PythonHighlighter(_RuleHighlighter):
    """Names and numbers by rule; then strings, comments and triple-quoted strings (which may span
    lines) in one left-to-right scan, so a ``#`` inside a string is not a comment and quotes
    inside a comment start nothing."""

    RULES = (
        (rf"\b(?:{_PY_BUILTINS})\b", "code_builtin", 0),
        (rf"\b(?:{_PY_KEYWORDS})\b", "code_keyword", 0),
        (r"\b(?:self|cls)\b", "code_self", 0),
        (r"\b(?:def|class)\s+(\w+)", "code_definition", 1),
        (r"\b[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][+-]?[0-9]+)?j?\b|\b0[xXoObB][0-9a-fA-F_]+\b",
         "code_number", 0),
        (r"^\s*@[\w.]+", "code_decorator", 0),
    )
    _SCAN = re.compile(r"""(?P<triple>[rRbBuUfF]{0,2}(?:\"\"\"|\'\'\'))"""
                       r"""|(?P<string>[rRbBuUfF]{0,2}(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'))"""
                       r"""|(?P<comment>\#.*)""")
    _CLOSE = {1: re.compile(r'"""'), 2: re.compile(r"'''")}

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt API)
        super().highlightBlock(text)
        self.setCurrentBlockState(0)
        string, comment = _style("code_string"), _style("code_comment", italic=True)
        position = 0
        state = self.previousBlockState()
        if state in self._CLOSE:  # inside a triple-quoted string from an earlier line
            end = self._CLOSE[state].search(text)
            if end is None:
                self.setFormat(0, len(text), string)
                self.setCurrentBlockState(state)
                return
            self.setFormat(0, end.end(), string)
            position = end.end()
        while (match := self._SCAN.search(text, position)) is not None:
            kind = match.lastgroup
            if kind == "comment":
                self.setFormat(match.start(), len(text) - match.start(), comment)
                return
            if kind == "string":
                self.setFormat(match.start(), match.end() - match.start(), string)
                position = match.end()
                continue
            state = 1 if match.group().endswith('"') else 2
            end = self._CLOSE[state].search(text, match.end())
            if end is None:
                self.setFormat(match.start(), len(text) - match.start(), string)
                self.setCurrentBlockState(state)
                return
            self.setFormat(match.start(), end.end() - match.start(), string)
            position = end.end()


class YamlHighlighter(_RuleHighlighter):
    RULES = (
        (r"^\s*-?\s*([\w.\-/ ]+?)\s*:(?=\s|$)", "code_keyword", 1),
        (r"\b(?:true|false|null|True|False|None|yes|no)\b", "code_builtin", 0),
        (r"(?<![\w.])-?[0-9][0-9_.eE+-]*\b", "code_number", 0),
        (r"\"[^\"\n]*\"|'[^'\n]*'", "code_string", 0),
        (r"(?:^|\s)#[^\n]*", "code_comment", 0),
    )


class JsonHighlighter(_RuleHighlighter):
    RULES = (
        (r"(?<![\w.])-?[0-9][0-9.eE+-]*\b", "code_number", 0),
        (r"\b(?:true|false|null)\b", "code_builtin", 0),
        (r"\"(?:[^\"\\\n]|\\.)*\"", "code_string", 0),
        (r"(\"(?:[^\"\\\n]|\\.)*\")\s*:", "code_keyword", 1),
    )


class MarkdownHighlighter(_RuleHighlighter):
    RULES = (
        (r"^#{1,6}\s.*$", "code_keyword", 0),
        (r"`[^`\n]+`", "code_string", 0),
        (r"\*\*[^*\n]+\*\*|__[^_\n]+__", "code_definition", 0),
        (r"^\s*(?:[-*+]|\d+\.)\s", "code_number", 0),
        (r"\[[^\]\n]+\]\([^)\n]+\)", "code_builtin", 0),
        (r"^```.*$", "code_comment", 0),
    )


class IniHighlighter(_RuleHighlighter):
    RULES = (
        (r"^\s*\[[^\]\n]+\]", "code_keyword", 0),
        (r"^\s*([\w.\-]+)\s*=", "code_builtin", 1),
        (r"(?<![\w.])-?[0-9][0-9_.]*\b", "code_number", 0),
        (r"\"[^\"\n]*\"|'[^'\n]*'", "code_string", 0),
        (r"(?:^|\s)[#;][^\n]*", "code_comment", 0),
    )


_BY_SUFFIX = {".py": PythonHighlighter, ".pyw": PythonHighlighter, ".pyi": PythonHighlighter,
              ".yaml": YamlHighlighter, ".yml": YamlHighlighter, ".json": JsonHighlighter,
              ".md": MarkdownHighlighter, ".toml": IniHighlighter, ".ini": IniHighlighter,
              ".cfg": IniHighlighter}


def highlighter_for(path: Path, document: QTextDocument) -> QSyntaxHighlighter | None:
    """A highlighter for ``path``'s kind of file, attached to ``document``; ``None`` = plain."""
    kind = _BY_SUFFIX.get(path.suffix.lower())
    return kind(document) if kind else None
