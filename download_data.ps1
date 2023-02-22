param ($dataversion)
$saname = 'nhpsa'

if ($dataversion -eq $null) {
  Write-Host "Missing argument: data version 
}

az storage blob download-batch `
  -d data `
  --pattern "$($dataversion)/synthetic/*" `
  -s data `
  --account-name nhpsa `
  --auth-mode login `
  --overwrite true

rm -rf data/synthetic
mv "data/$($dataversion)/synthetic" data
rmdir "data/$($dataversion)"
