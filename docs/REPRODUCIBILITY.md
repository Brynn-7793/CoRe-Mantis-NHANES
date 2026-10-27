# Reproducibility guide

## 1. Create labels and clinical covariates

Run once for each cycle, replacing the suffix and input directory as needed:

```bash
python scripts/prepare_nhanes_covariates.py \
  --raw-dir data/raw/2011_2012 \
  --output-dir data/derived/2011_2012 \
  --cycle-suffix G
```

Repeat for 2013–2014 with `--cycle-suffix H`.

## 2. Derive wearable features

Use the feature-extraction scripts with the corresponding minute-level
activity-monitor files. These scripts produce participant-level transparent
activity, rest–activity rhythm, and reliability features. Merge the derived
features with the covariate table by `SEQN` before model evaluation.

## 3. Run bidirectional cross-cycle evaluation

The main script requires one merged analytic table per cycle, a feature
dictionary, and an outcome-label table for each cycle:

```bash
python scripts/run_bidirectional_cross_cycle.py \
  --project . \
  --output-dir results/bidirectional \
  --activity-mode strict \
  --folds 5 \
  --bootstrap-iterations 2000 \
  --paired-iterations 10000 \
  --jobs 4
```

Set the project paths in the script configuration to the locations of the
derived tables before execution. The supplied defaults intentionally do not
point to participant-level files.

## 4. Sensitivity analysis and figures

Run the survey-weighted sensitivity analysis after the primary analysis:

```bash
python scripts/run_bidirectional_survey_sensitivity.py \
  --project . \
  --results-dir results/bidirectional \
  --output-dir results/survey_sensitivity
```

Then create the transportability figures:

```bash
python scripts/make_transportability_figures.py \
  --results-dir results/bidirectional \
  --output-dir figures
```

Exact outputs depend on the installed package versions, operating system, and
optional foundation-model feature availability. Record these details for any
new analysis run.
