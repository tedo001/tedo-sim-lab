"""Entry point of the installed app (cx_Freeze builds TEDO-AI-Lab.exe from this file): start the
lab's window with the command line it was given (``--workspace``, ``--smoke-test`` …)."""

import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
