"""Compatibility entry point for the original Wireshark capsa extraction."""

import sys

from rete_runner import main


if __name__ == "__main__":
    sys.exit(main(["--compile-commands", "/rete/wireshark/compile_commands.json",
                   "--contains", "capsa", *sys.argv[1:]]))
