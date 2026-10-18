"""Terminal modes, colours and replies, mixed into :class:`~core.terminal.screen.Screen`: SGR
(colours and styles), DEC private modes (the alternate screen, application cursor keys,
bracketed paste, mouse reporting, the cursor's visibility and shape), the replies programs ask
for (cursor position, device attributes, mode reports, colours), titles and character sets."""

from __future__ import annotations

from .cells import BLINK, BOLD, DEFAULT, DIM, HIDDEN, ITALIC, PLAIN, REVERSE, RGB, STRIKE, UNDERLINE

__all__ = ["ModesMixin"]

#: SGR codes that set a flag, and the codes that clear flags.
_SET = {1: BOLD, 2: DIM, 3: ITALIC, 4: UNDERLINE, 5: BLINK, 6: BLINK, 7: REVERSE, 8: HIDDEN,
        9: STRIKE, 21: UNDERLINE}
_CLEAR = {22: BOLD | DIM, 23: ITALIC, 24: UNDERLINE, 25: BLINK, 27: REVERSE, 28: HIDDEN,
          29: STRIKE}
#: DEC private modes kept as plain switches: mode number → attribute.
_SWITCHES = {1: "app_cursor", 5: "reverse_video", 7: "autowrap", 25: "cursor_visible",
             66: "app_keypad", 1004: "focus_events", 1006: "mouse_sgr", 1007: "alt_scroll",
             2004: "bracketed_paste", 2026: "synchronized"}
_MOUSE = (9, 1000, 1002, 1003)


class ModesMixin:
    """Uses the screen's ``attr``, cursor, ``responses`` and alternate-screen methods."""

    def reset_modes(self) -> None:
        self.autowrap = True
        self.origin_mode = False
        self.insert_mode = False
        self.newline_mode = False
        self.cursor_visible = True
        self.app_cursor = False
        self.app_keypad = False
        self.bracketed_paste = False
        self.reverse_video = False
        self.focus_events = False
        self.synchronized = False
        self.alt_scroll = True
        #: 0 = none; 9 (X10), 1000 (clicks), 1002 (drags), 1003 (any motion).
        self.mouse_mode = 0
        self.mouse_sgr = False
        #: DECSCUSR: 0/1/2 block, 3/4 underline, 5/6 bar.
        self.cursor_shape = 0

    # SGR ----------------------------------------------------------------------------------------
    def select_graphic_rendition(self, groups: list[list[int]]) -> None:
        """``groups``: the parameters, each with its ``:`` sub-parameters (``-1`` = empty)."""
        fg, bg, flags = self.attr
        i = 0
        while i < len(groups):
            group = groups[i]
            code = max(0, group[0])
            if code in (38, 48, 58):
                colour, i = _extended_colour(groups, i)
                if code == 38 and colour is not None:
                    fg = colour
                elif code == 48 and colour is not None:
                    bg = colour
            elif code == 0:
                fg, bg, flags = PLAIN
            elif code == 4 and len(group) > 1 and group[1] == 0:
                flags &= ~UNDERLINE  # 4:0 = no underline
            elif code in _SET:
                flags |= _SET[code]
            elif code in _CLEAR:
                flags &= ~_CLEAR[code]
            elif 30 <= code <= 37:
                fg = code - 30
            elif code == 39:
                fg = DEFAULT
            elif 40 <= code <= 47:
                bg = code - 40
            elif code == 49:
                bg = DEFAULT
            elif 90 <= code <= 97:
                fg = code - 90 + 8
            elif 100 <= code <= 107:
                bg = code - 100 + 8
            i += 1
        self.attr = (fg, bg, flags)

    # Modes ------------------------------------------------------------------------------------------
    def set_mode(self, modes: list[int], private: bool) -> None:
        for mode in modes:
            self._switch(mode, private, True)

    def reset_mode(self, modes: list[int], private: bool) -> None:
        for mode in modes:
            self._switch(mode, private, False)

    def _switch(self, mode: int, private: bool, on: bool) -> None:
        if not private:
            if mode == 4:
                self.insert_mode = on
            elif mode == 20:
                self.newline_mode = on
            return
        if mode in _SWITCHES:
            setattr(self, _SWITCHES[mode], on)
        elif mode == 6:
            self.origin_mode = on
            self.cursor_position(1, 1)
        elif mode in _MOUSE:
            self.mouse_mode = mode if on else 0
        elif mode in (47, 1047):
            if on:
                self.enter_alt_screen(clear=mode == 1047)
            else:
                self.leave_alt_screen()
        elif mode == 1048:
            (self.save_cursor if on else self.restore_cursor)()
        elif mode == 1049:
            if on:
                self.save_cursor()
                self.enter_alt_screen(clear=True)
                self.cursor_position(1, 1)
            else:
                self.leave_alt_screen()
                self.restore_cursor()

    def mode_state(self, mode: int, private: bool) -> int:
        """DECRQM's answer: 1 set, 2 reset, 0 not recognised."""
        if private:
            if mode in _SWITCHES:
                return 1 if getattr(self, _SWITCHES[mode]) else 2
            if mode == 6:
                return 1 if self.origin_mode else 2
            if mode in _MOUSE:
                return 1 if self.mouse_mode == mode else 2
            if mode in (47, 1047, 1049):
                return 1 if self.in_alt_screen else 2
            return 0
        if mode == 4:
            return 1 if self.insert_mode else 2
        if mode == 20:
            return 1 if self.newline_mode else 2
        return 0

    def report_mode(self, mode: int, private: bool) -> None:
        mark = "?" if private else ""
        self.responses.append(f"\x1b[{mark}{mode};{self.mode_state(mode, private)}$y")

    def soft_reset(self) -> None:
        """DECSTR: modes back to their defaults, the screen's content kept."""
        self.cursor_visible, self.origin_mode, self.autowrap = True, False, True
        self.insert_mode, self.app_cursor, self.app_keypad = False, False, False
        self.top, self.bottom = 0, self.rows - 1
        self.attr = PLAIN
        self.saved.clear()

    def set_cursor_style(self, style: int) -> None:
        self.cursor_shape = style if 0 <= style <= 6 else 0

    # Replies ---------------------------------------------------------------------------------------
    def device_status(self, request: int, private: bool = False) -> None:
        if request == 5:
            self.responses.append("\x1b[0n")
        elif request == 6:
            row = self.cy - (self.top if self.origin_mode else 0) + 1
            self.responses.append(f"\x1b[{'?' if private else ''}{row};{self.cx + 1}R")

    def device_attributes(self, kind: str = "") -> None:
        if kind == "":
            self.responses.append("\x1b[?62;22c")   # a VT220 with ANSI colour
        elif kind == ">":
            self.responses.append("\x1b[>1;10;0c")

    def window_op(self, params: list[int]) -> None:
        if params and params[0] == 18:  # the text area's size in characters
            self.responses.append(f"\x1b[8;{self.rows};{self.cols}t")

    def operating_system_command(self, command: str) -> None:
        """OSC: titles and colour queries; everything else (clipboard, hyperlinks) is ignored."""
        code, _, rest = command.partition(";")
        if code in ("0", "2"):
            self.title = rest[:256]
        elif code in ("10", "11", "12") and rest == "?":
            value = self.reported_colours[int(code)]
            red, green, blue = value[0:2], value[2:4], value[4:6]
            self.responses.append(f"\x1b]{code};rgb:{red}{red}/{green}{green}/{blue}{blue}\x1b\\")

    def designate_charset(self, slot: str, code: str) -> None:
        index = {"(": 0, ")": 1}.get(slot)
        if index is not None:
            self.charsets[index] = "0" if code == "0" else "B"

    def shift_out(self) -> None:
        self.shift = 1

    def shift_in(self) -> None:
        self.shift = 0


def _extended_colour(groups: list[list[int]], i: int) -> tuple[int | None, int]:
    """A 38/48/58 colour: ``38:5:n``, ``38:2::r:g:b`` or ``38;5;n``, ``38;2;r;g;b``.
    Returns (colour or None, index of the last parameter used)."""
    group = groups[i]
    if len(group) > 1:  # colon form, all in one parameter
        if group[1] == 5 and len(group) > 2:
            return _palette(group[2]), i
        if group[1] == 2 and len(group) >= 5:
            red, green, blue = group[-3:]
            return _rgb(red, green, blue), i
        return None, i
    following = [g[0] for g in groups[i + 1:i + 5]]
    if following[:1] == [5] and len(following) >= 2:
        return _palette(following[1]), i + 2
    if following[:1] == [2] and len(following) >= 4:
        return _rgb(*following[1:4]), i + 4
    return None, i


def _palette(index: int) -> int | None:
    return index if 0 <= index <= 255 else None


def _rgb(red: int, green: int, blue: int) -> int:
    channel = [min(255, max(0, value)) for value in (red, green, blue)]
    return RGB | (channel[0] << 16) | (channel[1] << 8) | channel[2]
