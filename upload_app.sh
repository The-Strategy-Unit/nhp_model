#/bin/sh
az storage blob upload-batch --account-name nhpdev --source model --destination app --destination-path model
az storage blob upload --account-name nhpdev -c app -f run_model.py 

conda pack -n nhp -o nhp.tar.gz
az storage blob upload --account-name nhpdev -c app -f nhp.tar.gz
rm nhp.tar.gz