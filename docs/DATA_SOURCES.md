# Public data sources

This workflow uses de-identified public NHANES data. Download the files from
the official CDC/NCHS pages; do not upload them to this repository.

## Required NHANES cycles

Create one directory per cycle, for example:

```text
data/
├── raw/
│   ├── 2011_2012/
│   └── 2013_2014/
└── derived/
```

For each cycle, download the demographic, depression questionnaire, medical
conditions, prescription medication, smoking, alcohol, body measures, and
physical activity monitor components referenced by
`scripts/prepare_nhanes_covariates.py` and the feature-extraction scripts.
The release suffix is `G` for 2011–2012 and `H` for 2013–2014.

The minute-level physical activity monitor files are `PAXMIN_G.XPT` and
`PAXMIN_H.XPT`; their matching monitor header files are `PAXHD_G.XPT` and
`PAXHD_H.XPT`.

## Outcome

The code derives the PHQ-9 total from the nine `DPQ` items and defines the
binary outcome as a total score of at least 10. Participants without complete
PHQ-9 data are excluded from outcome-specific analyses.

## Optional foundation-model features

The original analysis evaluated an optional foundation-model branch. Any
third-party checkpoint must be obtained from its original public source and
used according to its license. No checkpoint is redistributed here.
