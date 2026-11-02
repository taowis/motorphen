# motorphen

Motor phenotyping from markerless gait recordings. The aim is a small, open
implementation of a normative gait model: train on healthy walking, score each
joint by how far it departs from what the model expects, and reconstruct a
subject-specific normative trajectory. Joint-level outputs are meant to be
compared across genetically defined patient groups.

The model design follows the masked autoencoder approach described by
Motta et al. (GenGait, Artif Intell Med 2027, doi:10.1016/j.artmed.2026.103545).
Their code and data are not public; this is an independent reimplementation
from the published description, not their software.

## Status

Early setup. Nothing here is validated yet.

## Layout

- `src/motorphen/` package code
- `scripts/` data preparation and training runs
- `tests/` unit tests
- `docs/` design notes
- `data/` see `data/README.md`; raw data is not committed
- `results/` small summary outputs

No patient data goes into this repository.
