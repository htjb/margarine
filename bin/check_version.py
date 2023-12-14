#!/usr/bin/env python
"""Check Version.

Verify version has been incremented.

Based on check_version.py from the anesthetic GitHub repository:
https://github.com/handley-lab/anesthetic
"""


import sys
import subprocess
from packaging import version


# Filestructure
readme_file = "README.rst"


# Utility functions
def run_on_commandline(*args):
    """Run the given arguments as a command on the command line."""
    return subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def unit_incremented(version_a: str, version_b: str) -> bool:
    """Check if version_a is one version larger than version_b.

    Parameters
    ----------
    version_a: str
        New version, version number
    version_b: str
        Old version, version number

    Returns
    -------
    unit_incremented: bool
        Whether, version number has been unit incremented
    """
    # Convert to version objects
    version_a = version.parse(version_a)
    version_b = version.parse(version_b)

    # Check increment for pre-release versions
    if version_a.pre is not None and version_b.pre is not None:
        # Matching pre-release levels
        if version_a.pre[0] == version_b.pre[0]:
            return ((version_a.pre[1] == (int(version_b.pre[1]) + 1)) and
                    (version_a.base_version == version_b.base_version))
        # Differing pre-release levels
        else:
            return (version_a.pre[1] == 0 and
                    version_a.pre[0] > version_b.pre[0] and
                    version_a.base_version == version_b.base_version)

    # New pre-release level
    elif version_a.pre is not None:
        return (version_a.base_version > version_b.base_version and
                version_a.pre[1] == 0)

    # Full release
    elif version_b.pre is not None:
        return version_a.base_version == version_b.base_version

    # Standard version major, minor and micro increments
    else:
        return (version_a.micro == version_b.micro + 1 and
                version_a.minor == version_b.minor and
                version_a.major == version_b.major or
                version_a.micro == 0 and
                version_a.minor == version_b.minor + 1 and
                version_a.major == version_b.major or
                version_a.micro == 0 and
                version_a.minor == 0 and
                version_a.major == version_b.major+1)

def get_current_version() -> str:
    """Get current version of package from README.rst"""
    current_version = run_on_commandline("grep", ":Version:", readme_file)
    current_version = current_version.split(":")[-1].strip()
    return current_version

def main():
    """Check version is consistent and incremented correctly."""
    # Get current version from readme
    current_version = get_current_version()

    # Get previous version from main branch of code
    run_on_commandline("git", "fetch", "origin", "master")
    readme_contents = run_on_commandline("git", "show",
                                         "remotes/origin/master:" +
                                         readme_file)

    previous_version = None
    for line in readme_contents.splitlines():
        if ":Version:" in line:
            previous_version = line.split(":")[-1].strip()
            break

    if previous_version is None:
        raise ValueError("Could not find version in README.rst on master branch")

    # Check versions have been incremented
    if not unit_incremented(current_version, previous_version):
        sys.stderr.write(("Version must be incremented by one:\n"
                          "HEAD:   {},\n"
                          "master: {}.\n").format(current_version,
                                                  previous_version))
        sys.exit(1)

    # No issues found, exit happily :)
    sys.exit(0)


if __name__ == "__main__":
    main()
