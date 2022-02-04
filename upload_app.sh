#/bin/sh
az storage blob upload --account-name nhpdev -c app -f model.py 
az storage blob upload --account-name nhpdev -c app -f requirements.txt