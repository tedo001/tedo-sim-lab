"""The terminal emulator: what programs print lands where xterm would put it."""

from __future__ import annotations

import pytest

from core.terminal import Parser, Screen
from core.terminal.cells import BOLD, DEFAULT, REVERSE, RGB, UNDERLINE


def term(cols: int = 10, rows: int = 4, scrollback: int = 100) -> tuple[Screen, Parser]:
    screen = Screen(cols, rows, scrollback)
    return screen, Parser(screen)


def text(screen: Screen) -> list[str]:
    return [line.rstrip() for line in screen.display()]


def test_printing_wraps_only_when_the_next_character_arrives() -> None:
    screen, parser = term(5, 3)
    parser.feed("abcde")
    assert text(screen) == ["abcde", "", ""] and (screen.cx, screen.cy) == (4, 0) and screen.wrap_next
    parser.feed("f")
    assert text(screen) == ["abcde", "f", ""] and screen.lines[0].wrapped
    parser.feed("\x1b[?7l" + "ghijklmn")  # autowrap off: the overflow overwrites the last column
    assert text(screen)[1] == "fghin"


def test_controls_cr_lf_backspace_and_tabs() -> None:
    screen, parser = term(20, 3)
    parser.feed("ab\bX\r\nc\td\tE")
    assert text(screen)[:2] == ["aX", "c       d       E"]
    parser.feed("\x1b[3g\x1b[1;4H\x1bH\r\tZ")  # clear all stops, set one at the 4th column
    assert screen.cx == 4 and text(screen)[0] == "aX Z"


def test_cursor_moves_clamp_to_the_screen() -> None:
    screen, parser = term(10, 5)
    parser.feed("\x1b[3;4H")
    assert (screen.cx, screen.cy) == (3, 2)
    parser.feed("\x1b[10A\x1b[99C")
    assert (screen.cx, screen.cy) == (9, 0)
    parser.feed("\x1b[2B\x1b[3D\x1b[7G\x1b[4d")
    assert (screen.cx, screen.cy) == (6, 3)
    parser.feed("\x1b[H\x1b[2E")
    assert (screen.cx, screen.cy) == (0, 2)


def test_erasing_uses_the_current_background() -> None:
    screen, parser = term(6, 3)
    parser.feed("aaaaaa\r\nbbbbbb\r\ncccccc\x1b[2;3H\x1b[44m\x1b[K")
    assert text(screen) == ["aaaaaa", "bb", "cccccc"]
    assert screen.lines[1][4][1][1] == 4  # erased cells take the blue background
    parser.feed("\x1b[1K")
    assert text(screen)[1] == ""
    parser.feed("\x1b[0m\x1b[J")
    assert text(screen) == ["aaaaaa", "", ""]
    parser.feed("\x1b[1;4H\x1b[1J")
    assert text(screen)[0] == "    aa"
    parser.feed("\x1b[2J")
    assert text(screen) == ["", "", ""]


def test_inserting_and_deleting_characters_and_lines() -> None:
    screen, parser = term(6, 4)
    parser.feed("abcdef\x1b[1;3H\x1b[2@")
    assert text(screen)[0] == "ab  cd"
    parser.feed("\x1b[3P")
    assert text(screen)[0] == "abd"
    parser.feed("\x1b[1;1H\x1b[2X")
    assert text(screen)[0] == "  d"
    parser.feed("\x1b[2J\x1b[H1\r\n2\r\n3\r\n4\x1b[2;1H\x1b[L")
    assert text(screen) == ["1", "", "2", "3"]
    parser.feed("\x1b[2M")
    assert text(screen) == ["1", "3", "", ""]


def test_scroll_regions_and_the_scrollback() -> None:
    screen, parser = term(4, 4)
    parser.feed("1\r\n2\r\n3\r\n4\r\n5")
    assert text(screen) == ["2", "3", "4", "5"] and [s[0][0] for s in screen.scrollback] == ["1"]
    parser.feed("\x1b[2;3r")  # lines 2-3 scroll; the cursor goes home
    assert (screen.cx, screen.cy, screen.top, screen.bottom) == (0, 0, 1, 2)
    parser.feed("\x1b[3;1H\nX")
    assert text(screen) == ["2", "4", "X", "5"] and len(screen.scrollback) == 1  # a region never feeds it
    parser.feed("\x1b[2;1H\x1bMY")
    assert text(screen) == ["2", "Y", "4", "5"]
    parser.feed("\x1b[r\x1b[2S")
    assert text(screen)[:2] == ["4", "5"] and len(screen.scrollback) == 3
    parser.feed("\x1b[T")
    assert text(screen)[:2] == ["", "4"]


def test_origin_mode_addresses_inside_the_region() -> None:
    screen, parser = term(5, 6)
    parser.feed("\x1b[3;5r\x1b[?6h\x1b[1;1H")
    assert screen.cy == 2
    parser.feed("\x1b[9;1H")
    assert screen.cy == 4
    parser.feed("\x1b[6n")
    assert screen.responses[-1] == "\x1b[3;1R"


def test_the_alternate_screen_keeps_the_shell_intact() -> None:
    screen, parser = term(8, 3)
    parser.feed("prompt$ \x1b[?1049h\x1b[Hvim here")
    assert screen.in_alt_screen and text(screen)[0] == "vim here"
    for _ in range(5):
        parser.feed("\r\n")
    assert len(screen.scrollback) == 0  # full-screen programs do not fill the scrollback
    parser.feed("\x1b[?1049l")
    assert not screen.in_alt_screen and text(screen)[0] == "prompt$" and screen.cx == 7


def test_colours_and_styles() -> None:
    screen, parser = term(20, 2)
    parser.feed("\x1b[1;4;31;44mA\x1b[0mB\x1b[38;5;208mC\x1b[38;2;10;20;30mD\x1b[48:2::1:2:3mE"
                "\x1b[7;39mF\x1b[22;24;27m\x1b[92;103mG\x1b[mH")
    cells = screen.lines[0]
    assert cells[0][1] == (1, 4, BOLD | UNDERLINE)
    assert cells[1][1] == (DEFAULT, DEFAULT, 0)
    assert cells[2][1][0] == 208
    assert cells[3][1][0] == RGB | (10 << 16) | (20 << 8) | 30
    assert cells[4][1][1] == RGB | (1 << 16) | (2 << 8) | 3
    assert cells[5][1][0] == DEFAULT and cells[5][1][2] & REVERSE
    assert cells[6][1] == (10, 11, 0) and cells[7][1] == (DEFAULT, DEFAULT, 0)


def test_wide_and_combining_characters() -> None:
    screen, parser = term(5, 2)
    parser.feed("a漢b")
    assert [c[0] for c in screen.lines[0][:4]] == ["a", "漢", "", "b"] and screen.cx == 4
    parser.feed("字")  # no room at the last column: it wraps whole
    assert screen.lines[0][4][0] == " " and screen.lines[1][0][0] == "字"
    parser.feed("é")
    assert screen.lines[1][2][0] == "é"


def test_line_drawing_charset_and_insert_mode_and_repeat() -> None:
    screen, parser = term(10, 2)
    parser.feed("\x1b(0lqk\x1b(Bx")
    assert text(screen)[0] == "┌─┐x"
    parser.feed("\r\x1b[4hZ\x1b[4l")
    assert text(screen)[0] == "Z┌─┐x"
    parser.feed("\r\n-\x1b[3b")
    assert text(screen)[1] == "----"


def test_replies_to_queries() -> None:
    screen, parser = term(30, 10)
    parser.feed("\x1b[5n\x1b[c\x1b[>c\x1b]11;?\x07\x1b]10;?\x1b\\\x1b[18t\x1b[?2004h\x1b[?2004$p\x1b[?9999$p")
    assert screen.responses == ["\x1b[0n", "\x1b[?62;22c", "\x1b[>1;10;0c",
                                "\x1b]11;rgb:0f0f/1111/1414\x1b\\", "\x1b]10;rgb:e3e3/e6e6/eaea\x1b\\",
                                "\x1b[8;10;30t", "\x1b[?2004;1$y", "\x1b[?9999;0$y"]


def test_sequences_split_over_reads_and_garbage_never_print() -> None:
    screen, parser = term(30, 3)
    for char in "\x1b]0;my title\x1b\\\x1b[1;31mred\x1b[0m \x1bP+q544e\x1b\\ok\x1b[?25l\x1b[!p":
        parser.feed(char)
    assert screen.title == "my title" and text(screen)[0] == "red ok" and screen.cursor_visible
    parser.feed("\x1b[12;34;56;78~\x1b[?1;2;3X\x1b%G\x1b[" + "9" * 400 + "mz")
    assert text(screen)[0] == "red okz"


def test_modes_the_widget_reads() -> None:
    screen, parser = term()
    parser.feed("\x1b[?1h\x1b[?2004h\x1b[?1002h\x1b[?1006h\x1b[?1004h\x1b[5 q\x1b[?25l")
    assert screen.app_cursor and screen.bracketed_paste and screen.mouse_mode == 1002
    assert screen.mouse_sgr and screen.focus_events and screen.cursor_shape == 5 and not screen.cursor_visible
    parser.feed("\x1bc")
    assert not screen.app_cursor and screen.mouse_mode == 0 and screen.cursor_visible


def test_save_and_restore_the_cursor() -> None:
    screen, parser = term(10, 5)
    parser.feed("\x1b[3;4H\x1b[1m\x1b7\x1b[H\x1b[0mx\x1b8y")
    assert (screen.cy, screen.cx) == (2, 4) and screen.lines[2][3] == ("y", (DEFAULT, DEFAULT, BOLD))


@pytest.mark.parametrize("rows", [2, 3])
def test_resizing_keeps_the_cursor_line_and_uses_the_scrollback(rows: int) -> None:
    screen, parser = term(6, 5)
    parser.feed("1\r\n2\r\n3\r\n4\r\n5")
    screen.resize(rows, 4)
    assert text(screen)[-1] == "5" and screen.cy == rows - 1 and screen.cols == 4
    assert [s[0][0] for s in screen.scrollback] == ["1", "2", "3", "4"][: 5 - rows]
    screen.resize(5, 8)
    assert text(screen) == ["1", "2", "3", "4", "5"] and screen.cy == 4 and not screen.scrollback


def test_resizing_a_short_screen_drops_blank_lines_below_the_cursor() -> None:
    screen, parser = term(6, 6)
    parser.feed("prompt")
    screen.resize(3, 6)
    assert text(screen)[0] == "prompt" and screen.cy == 0 and not screen.scrollback


def test_history_numbering_survives_scrolling() -> None:
    screen, parser = term(4, 2, scrollback=3)
    for number in range(6):
        parser.feed(f"{number}\r\n")
    first = screen.history_total - len(screen.scrollback)
    lines = range(first, screen.history_total + 2)
    assert [screen.history_line(n)[0][0] for n in lines] == ["2", "3", "4", "5", " "]
    assert screen.history_line(first - 1) is None
