#/bin/sh
az storage blob upload --account-name nhpdev -c app -f model.py 
conda list -e | awk 'BEGIN { FS = \"=\" }; !/^(python=|#)/ { print $1\" == \"$2 }' > requirements.txt
az storage blob upload --account-name nhpdev -c app -f requirements.txt