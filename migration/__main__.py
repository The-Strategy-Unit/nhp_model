"""Migration scripts
scripts for migrating between versions of the app
"""

import sys

from migration.v04 import v03_to_v04

print(sys.argv)
v03_to_v04(sys.argv[1])
