#/bin/sh
az storage blob upload-batch --account-name nhpdev --source model --destination app --destination-path model
az storage blob upload --account-name nhpdev -c app -f run_model.py 

# in the environment.yml file, make sure that dependencies: starts on line 4,
# and the first item listed is python. we can then skip the first 5 lines to
# extract the names of the packages, then we can use conda list to find out
# exactly what version of the packages we are using
awk '(NR>5) { gsub(/^\s+-\s+|=.*$/, """"); print $0 }' environment.yml > /tmp/.requirements &&
  conda list --name=nhp |
  awk '{print $1 "==" $2}' |
  grep -F -f /tmp/.requirements > requirements.txt; # for whatever reason, using requirements.txt as -f caused a NULL output
  rm /tmp/.requirements
# az storage blob upload --account-name nhpdev -c app -f requirements.txt
pip install -r requirements.txt --target=test/pip
az storage blob upload-batch --account-name nhpdev --source test/pip --destination app --destination-path site-packages