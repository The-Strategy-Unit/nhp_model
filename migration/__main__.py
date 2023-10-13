"""Migration scripts
scripts for migrating between versions of the app
"""

import sys

import migration.v04 as v04
import migration.v05 as v05


def main(version, path):
    match version:
        case "v03":
            v04.convert_all_files_in_folder(path)
        case "v04":
            v05.convert_all_files_in_folder(path)
        case _:
            print("Unknown version")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
