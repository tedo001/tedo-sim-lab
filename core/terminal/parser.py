"""Read a program's output (text, as decoded from the terminal) and apply it to a
:class:`~core.terminal.screen.Screen`: printable runs, C0 controls, and ESC, CSI, OSC and
DCS sequences, the way xterm reads them. Unknown sequences are skipped whole, never printed.

The parser keeps its state between calls, so a sequence may arrive split over several reads.
"""

from __future__ import annotations

import re

from .screen import Screen

__all__ = ["Parser"]

#: Anything that is not printed as is: C0 controls, DEL, and C1 code points.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
GROUND, ESCAPE, ESCAPE_INTERMEDIATE, CSI, CSI_IGNORE, OSC, STRING = range(7)
#: Longest CSI parameter list and OSC/DCS string kept; longer ones are dropped (garbage, or an
#: attempt to make the terminal hold a lot of memory).
MAX_PARAMS = 256
MAX_STRING = 8192


class Parser:
    def __init__(self, screen: Screen) -> None:
        self.screen = screen
        self.state = GROUND
        self._buffer = ""        # CSI parameters / OSC string so far
        self._private = ""       # CSI private marker: ? > = <
        self._intermediate = ""  # ESC / CSI intermediate characters
        self._string_escape = False

    def feed(self, text: str) -> None:
        screen, i, n = self.screen, 0, len(text)
        while i < n:
            if self.state == GROUND:
                match = _CONTROL.search(text, i)
                if match is None:
                    screen.draw(text[i:])
                    return
                start = match.start()
                if start > i:
                    screen.draw(text[i:start])
                self._control(text[start])
                i = start + 1
                continue
            char = text[i]
            i += 1
            if self.state in (OSC, STRING):
                self._string(char)
            elif char == "\x1b":
                self._begin_escape()
            elif char in "\x18\x1a":  # CAN, SUB: abort the sequence
                self.state = GROUND
            elif char < " " or char == "\x7f":
                self._control(char)  # controls inside a sequence still act
            elif self.state == ESCAPE:
                self._escape(char)
            elif self.state == ESCAPE_INTERMEDIATE:
                self._escape_intermediate(char)
            elif self.state == CSI:
                self._csi(char)
            elif self.state == CSI_IGNORE and "@" <= char <= "~":
                self.state = GROUND  # the end of a sequence too long to keep

    # Controls and ESC --------------------------------------------------------------------------
    def _control(self, char: str) -> None:
        screen = self.screen
        if char == "\x1b":
            self._begin_escape()
        elif char in "\n\x0b\x0c":
            screen.linefeed()
        elif char == "\r":
            screen.carriage_return()
        elif char == "\x08":
            screen.backspace()
        elif char == "\t":
            screen.tab()
        elif char == "\x07":
            screen.bell()
        elif char == "\x0e":
            screen.shift_out()
        elif char == "\x0f":
            screen.shift_in()

    def _begin_escape(self) -> None:
        self.state = ESCAPE
        self._intermediate = ""

    def _escape(self, char: str) -> None:
        screen = self.screen
        self.state = GROUND
        if char == "[":
            self.state, self._buffer, self._private, self._intermediate = CSI, "", "", ""
        elif char == "]":
            self.state, self._buffer, self._string_escape = OSC, "", False
        elif char in "PX^_":  # DCS, SOS, PM, APC: read to the end and ignore
            self.state, self._buffer, self._string_escape = STRING, "", False
        elif char in "()*+#% ":
            self.state, self._intermediate = ESCAPE_INTERMEDIATE, char
        elif char == "7":
            screen.save_cursor()
        elif char == "8":
            screen.restore_cursor()
        elif char == "D":
            screen.index()
        elif char == "E":
            screen.next_line()
        elif char == "H":
            screen.set_tab_stop()
        elif char == "M":
            screen.reverse_index()
        elif char == "c":
            screen.reset()
        elif char == "=":
            screen.app_keypad = True
        elif char == ">":
            screen.app_keypad = False
        elif char == "Z":
            screen.device_attributes()

    def _escape_intermediate(self, char: str) -> None:
        self.state = GROUND
        if self._intermediate in "()*+":
            self.screen.designate_charset(self._intermediate, char)
        elif self._intermediate == "#" and char == "8":
            self.screen.alignment_test()

    # Strings (OSC, DCS, ...) ----------------------------------------------------------------------
    def _string(self, char: str) -> None:
        if self._string_escape:
            self._string_escape = False
            if char == "\\":  # ST
                self._end_string()
                return
            self._end_string()
            self._begin_escape()
            self._escape(char)
            return
        if char == "\x1b":
            self._string_escape = True
        elif char == "\x07" and self.state == OSC:
            self._end_string()
        elif char in "\x18\x1a":
            self.state = GROUND
        elif len(self._buffer) < MAX_STRING:
            self._buffer += char

    def _end_string(self) -> None:
        if self.state == OSC:
            self.screen.operating_system_command(self._buffer)
        self.state, self._buffer = GROUND, ""

    # CSI ---------------------------------------------------------------------------------------------
    def _csi(self, char: str) -> None:
        if char in "<=>?" and not self._buffer and not self._private:
            self._private = char
        elif char in "0123456789;:":
            self._buffer += char
            if len(self._buffer) > MAX_PARAMS:
                self.state = CSI_IGNORE
        elif " " <= char <= "/":
            self._intermediate += char
        elif "@" <= char <= "~":
            self.state = GROUND
            self._dispatch(char)
        else:
            self.state = GROUND

    def _dispatch(self, final: str) -> None:
        screen, private, inter = self.screen, self._private, self._intermediate
        raw = self._buffer
        params = [_number(part.split(":")[0]) for part in raw.split(";")] if raw else []
        first = params[0] if params else 0
        if inter == " " and final == "q":
            screen.set_cursor_style(first)
        elif inter == "!" and final == "p":
            screen.soft_reset()
        elif inter == "$" and final == "p":
            screen.report_mode(first, private == "?")
        elif inter:
            return
        elif private == "?":
            if final == "h":
                screen.set_mode(params, True)
            elif final == "l":
                screen.reset_mode(params, True)
            elif final == "n":
                screen.device_status(first, private=True)
            elif final == "J":
                screen.erase_display(first)
            elif final == "K":
                screen.erase_line(first)
        elif private == ">":
            if final == "c":
                screen.device_attributes(">")
        elif not private:
            handler = _CSI.get(final)
            if handler is not None:
                handler(screen, params, raw)


def _number(text: str) -> int:
    return int(text) if text.isdigit() else 0


def _arg(params: list[int], index: int = 0, default: int = 1) -> int:
    value = params[index] if index < len(params) else 0
    return value or default


def _sgr(screen: Screen, _params: list[int], raw: str) -> None:
    groups = [[int(x) if x.isdigit() else -1 for x in part.split(":")] for part in raw.split(";")] \
        if raw else [[0]]
    screen.select_graphic_rendition(groups)


def _scroll_down(screen: Screen, params: list[int], _raw: str) -> None:
    if len(params) <= 1:  # CSI T with five parameters is mouse tracking, not scrolling
        screen.scroll_down(_arg(params))


_CSI = {
    "@": lambda s, p, r: s.insert_chars(_arg(p)),
    "A": lambda s, p, r: s.cursor_up(_arg(p)),
    "B": lambda s, p, r: s.cursor_down(_arg(p)),
    "C": lambda s, p, r: s.cursor_forward(_arg(p)),
    "D": lambda s, p, r: s.cursor_back(_arg(p)),
    "E": lambda s, p, r: (s.cursor_down(_arg(p)), s.carriage_return()),
    "F": lambda s, p, r: (s.cursor_up(_arg(p)), s.carriage_return()),
    "G": lambda s, p, r: s.cursor_column(_arg(p)),
    "H": lambda s, p, r: s.cursor_position(_arg(p, 0), _arg(p, 1)),
    "I": lambda s, p, r: s.tab(_arg(p)),
    "J": lambda s, p, r: s.erase_display(_arg(p, default=0)),
    "K": lambda s, p, r: s.erase_line(_arg(p, default=0)),
    "L": lambda s, p, r: s.insert_lines(_arg(p)),
    "M": lambda s, p, r: s.delete_lines(_arg(p)),
    "P": lambda s, p, r: s.delete_chars(_arg(p)),
    "S": lambda s, p, r: s.scroll_up(_arg(p)),
    "T": _scroll_down,
    "X": lambda s, p, r: s.erase_chars(_arg(p)),
    "Z": lambda s, p, r: s.back_tab(_arg(p)),
    "`": lambda s, p, r: s.cursor_column(_arg(p)),
    "a": lambda s, p, r: s.cursor_forward(_arg(p)),
    "b": lambda s, p, r: s.repeat_last(_arg(p)),
    "c": lambda s, p, r: s.device_attributes() if _arg(p, default=0) == 0 else None,
    "d": lambda s, p, r: s.cursor_row(_arg(p)),
    "e": lambda s, p, r: s.cursor_down(_arg(p)),
    "f": lambda s, p, r: s.cursor_position(_arg(p, 0), _arg(p, 1)),
    "g": lambda s, p, r: s.clear_tab_stop(_arg(p, default=0)),
    "h": lambda s, p, r: s.set_mode(p, False),
    "l": lambda s, p, r: s.reset_mode(p, False),
    "m": _sgr,
    "n": lambda s, p, r: s.device_status(_arg(p, default=0)),
    "r": lambda s, p, r: s.set_scroll_region(_arg(p, 0, 0), _arg(p, 1, 0)),
    "s": lambda s, p, r: s.save_cursor(),
    "t": lambda s, p, r: s.window_op(p),
    "u": lambda s, p, r: s.restore_cursor(),
}
