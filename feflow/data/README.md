# FEflow Data files

This directory contains the data files for `feflow`. It is used both for testing, benchmarking or
running examples that use the different routines in the package. They are left in this directory as
a convenient way to ship them alongside the code, as long as they are not many files and the sizes
are not more than a few MBs.

You can programatically access our data files using `importlib` with something similar to

```python
from importlib.resources import files
data_text = files('feflow.data.subdir').joinpath('datafile.ext').read_text()
```

Please adapt as needed.

## Host-guest data

Host-guest data from SAMPL challenge with curcubit uril (CB7) host and two of its guests. The files
can be accessed from the `hist-guest` subdirectory.

For convenience the format of these molecules are in serialized `gufe.SmallMoleculeComponent` `json`
files, which contain all the cheminformatics and force field information of the molecules to be
readily used.

Please refer to `feflow/tests/test_sanity.py::test_roundtrip_charge_transformation` for an example
on how to use this data within `feflow`.

