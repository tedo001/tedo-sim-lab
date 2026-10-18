"""Keys → the bytes a terminal program expects (xterm conventions): arrows and the editing keys
(application cursor mode included), function keys, Ctrl and Alt combinations, Enter, Tab and
Backspace. ``encode`` returns ``""`` for keys the terminal does not send (plain modifiers)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent

from core.terminal import Screen

__all__ = ["encode", "modifier_code"]

K = Qt.Key
_CURSOR = {K.Key_Up: "A", K.Key_Down: "B", K.Key_Right: "C", K.Key_Left: "D",
           K.Key_Home: "H", K.Key_End: "F"}
_TILDE = {K.Key_Insert: 2, K.Key_Delete: 3, K.Key_PageUp: 5, K.Key_PageDown: 6,
          K.Key_F5: 15, K.Key_F6: 17, K.Key_F7: 18, K.Key_F8: 19, K.Key_F9: 20, K.Key_F10: 21,
          K.Key_F11: 23, K.Key_F12: 24}
_SS3_F = {K.Key_F1: "P", K.Key_F2: "Q", K.Key_F3: "R", K.Key_F4: "S"}
#: Ctrl + punctuation → the C0 control xterm sends.
_CTRL_PUNCT = {K.Key_Space: "\x00", K.Key_At: "\x00", K.Key_2: "\x00", K.Key_BracketLeft: "\x1b",
               K.Key_3: "\x1b", K.Key_Backslash: "\x1c", K.Key_4: "\x1c", K.Key_BracketRight: "\x1d",
               K.Key_5: "\x1d", K.Key_AsciiCircum: "\x1e", K.Key_6: "\x1e", K.Key_Underscore: "\x1f",
               K.Key_Minus: "\x1f", K.Key_Slash: "\x1f", K.Key_7: "\x1f", K.Key_8: "\x7f",
               K.Key_Question: "\x7f"}


def modifier_code(modifiers: Qt.KeyboardModifier) -> int:
    """xterm's modifier parameter: 1 + Shift(1) + Alt(2) + Ctrl(4); 1 = none."""
    code = 1
    if modifiers & Qt.KeyboardModifier.ShiftModifier:
        code += 1
    if modifiers & Qt.KeyboardModifier.AltModifier:
        code += 2
    if modifiers & Qt.KeyboardModifier.ControlModifier:
        code += 4
    return code


def encode(event: QKeyEvent, screen: Screen) -> str:
    key, mods = event.key(), event.modifiers()
    ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
    alt = bool(mods & Qt.KeyboardModifier.AltModifier)
    shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
    code = modifier_code(mods)

    if key in _CURSOR:
        final = _CURSOR[key]
        if code > 1:
            return f"\x1b[1;{code}{final}"
        return ("\x1bO" if screen.app_cursor else "\x1b[") + final
    if key in _TILDE:
        number = _TILDE[key]
        return f"\x1b[{number};{code}~" if code > 1 else f"\x1b[{number}~"
    if key in _SS3_F:
        final = _SS3_F[key]
        return f"\x1b[1;{code}{final}" if code > 1 else "\x1bO" + final
    if key in (K.Key_Return, K.Key_Enter):
        text = "\r\n" if screen.newline_mode else "\r"
        return ("\x1b" + text) if alt else text
    if key == K.Key_Backspace:
        text = "\x08" if ctrl else "\x7f"
        return ("\x1b" + text) if alt else text
    if key == K.Key_Tab:
        return "\x1b[Z" if shift else "\t"
    if key == K.Key_Backtab:
        return "\x1b[Z"
    if key == K.Key_Escape:
        return "\x1b"

    text = event.text()
    if ctrl and alt and text and text.isprintable():
        return text  # AltGr, which Windows reports as Ctrl+Alt, typed a character
    if ctrl:  # Ctrl+letter → C0 control; with Alt, prefixed by ESC
        control = ""
        if K.Key_A <= key <= K.Key_Z:
            control = chr(key - K.Key_A + 1)
        elif key in _CTRL_PUNCT:
            control = _CTRL_PUNCT[key]
        return ("\x1b" + control) if control and alt else control
    if not text:
        return ""
    return ("\x1b" + text) if alt else text
