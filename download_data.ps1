param (
  $dataversion = "dev",
  $dataset = "synthetic",
  $year = "2019"
)

$saname = 'nhpsa'

az storage blob download-batch `
  -d data `
  --pattern "$dataversion/$year/$dataset/*" `
  -s data `
  --account-name nhpsa `
  --auth-mode login `
  --overwrite true

Remove-Item -Recurse -Force "data/$year/$dataset"
Move-Item "data/$dataversion/$year/$dataset" "data/$year/$dataset"
Remove-Item -Recurse -Force "data/$dataversion"
