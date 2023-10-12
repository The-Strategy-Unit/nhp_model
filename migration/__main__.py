"""Migration scripts
scripts for migrating between versions of the app
"""

import sys

from migration.v04 import v03_to_v04
from migration.v05 import v04_to_v05

print(sys.argv)

match sys.argv[1]:
    case "v03":
        v03_to_v04(sys.argv[2])
    case "v04":
        v04_to_v05(sys.argv[2])
    case _:
        print("Unknown migration type")
