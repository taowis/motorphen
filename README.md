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

The reimplementation has been trained on 110 able-bodied adults from a public
motion capture dataset (Van Criekinge et al., Sci Data 2023) and evaluated on
held-out healthy adults and on 50 stroke survivors whose paretic side is known.
Results are exploratory and come from our implementation of the published
description, with marker-based rather than markerless input.

- E1, specificity in held-out healthy adults: `results/e1/README.md`
- E2, paretic side in stroke survivors: `results/e2/README.md`
- E3, inference choices, age range of the reference, window length and
  training size: `results/e3/`
- Short summary of E1 and E2: `docs/summary-e1-e2.md`

Details the paper leaves open are our choices, listed in
`docs/e1-implementation-spec.md`.

## Layout

- `src/motorphen/` package code
- `scripts/` data preparation and training runs
- `tests/` unit tests
- `docs/` design notes
- `data/` see `data/README.md`; raw data is not committed
- `results/` small summary outputs

No patient data goes into this repository.

## License

Code is released under the MIT License (`LICENSE`). The gait data are not part of
this repository; see `data/README.md` for their source and licence.
