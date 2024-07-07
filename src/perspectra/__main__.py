#! /usr/bin/env python3

"""
The main entry point. Invoke as `perspectra`.
"""

import sys


def main():
    from perspectra import cli
    cli.execute_arguments(sys.argv[1:])

if __name__ == '__main__':
    main()
