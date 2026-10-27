# CoRe-Mantis-NHANES

Reproducibility code for the analysis of cross-cycle transportability of
depression-screening models using NHANES wrist-worn activity data.

## Scope

The repository contains the analysis workflow used to construct covariates,
derive wearable features, train the prespecified model branches, and evaluate
bidirectional transportability between the 2011–2012 and 2013–2014 NHANES
cycles. It does not contain participant-level data, trained model weights,
predictions, or manuscript files.

The primary outcome is clinically relevant depressive symptoms, defined as a
PHQ-9 total score of 10 or higher among participants with complete PHQ-9 data.

## Data access

All source data are public and must be downloaded directly from the National
Center for Health Statistics before running the workflow. This repository does
not redistribute NHANES files.

- [NHANES 2011–2012](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011)
- [NHANES 2013–2014](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013)
- [Physical Activity Monitor documentation](https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/PAXMIN_G.htm)

The required public components and the expected directory layout are described
in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Environment

Create a Python 3.11 environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The main transportability analysis is CPU-compatible. Optional extraction of
foundation-model embeddings requires PyTorch and the corresponding public
checkpoint, which are not included here.

## Reproduction workflow

1. Download the public NHANES files into the two cycle-specific directories.
2. Build labels and clinical covariates with `prepare_nhanes_covariates.py`.
3. Derive transparent activity and rest–activity rhythm features using the
   preprocessing scripts in `scripts/`.
4. Assemble the merged analytic table for each cycle.
5. Run `run_bidirectional_cross_cycle.py` to freeze the development-cycle
   workflow and evaluate it in the other cycle.
6. Run `run_bidirectional_survey_sensitivity.py` and
   `make_transportability_figures.py` for the reported sensitivity analyses and
   figures.

Command templates and required inputs are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Software and data use

Please cite NHANES according to the National Center for Health Statistics data
use guidance. The analysis code is released under the MIT License. A manuscript
citation will be added after publication.
