#!/usr/bin/env python
"""Get Version.

This script will print the version of the current project.
"""


import sys
from check_version import run_on_commandline, readme_file, get_current_version


def main():
    """Print the current version of the project."""
    # Get current version from readme
    current_version = get_current_version()

    # Print current version
    sys.stdout.write(f"{current_version}")


if __name__ == '__main__':
    main()
