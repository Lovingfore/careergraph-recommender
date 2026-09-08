# Raw O*NET source data

The repository intentionally does not commit the full O*NET text extract so
that clones stay lightweight. Download the O*NET 30.2 Database text release
from the official source:

<https://www.onetcenter.org/database.html>

Extract the archive so that `db_30_2_text/Occupation Data.txt` and
`db_30_2_text/Skills.txt` are available under this directory. Then run:

```powershell
python src/prepare_dataset.py
```

The selected O*NET occupation and skill rows are public source data under the
source license. The generated user skill events and job transitions in
`data/clean/` are deterministic teaching data, not real survey records.
