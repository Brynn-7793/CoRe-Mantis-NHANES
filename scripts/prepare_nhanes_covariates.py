"""Create an auditable NHANES covariate table for depression models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PHQ_ITEMS = [f"DPQ0{number}" for number in range(10, 100, 10)]
CHRONIC_ITEMS = [
    "MCQ160A",
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F",
    "MCQ160G",
    "MCQ160K",
    "MCQ160L",
    "MCQ160M",
    "MCQ160N",
    "MCQ160O",
]
ANTIDEPRESSANT_TERMS = (
    "amitriptyline|amoxapine|bupropion|citalopram|clomipramine|desipramine|"
    "desvenlafaxine|doxepin|duloxetine|escitalopram|fluoxetine|fluvoxamine|"
    "imipramine|maprotiline|mirtazapine|nefazodone|nortriptyline|paroxetine|"
    "protriptyline|sertraline|tranylcypromine|trazodone|trimipramine|venlafaxine|"
    "vortioxetine"
)
CYCLE_SUFFIX = "H"


def cycle_file(component: str) -> str:
    return f"{component}_{CYCLE_SUFFIX}.XPT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cycle-suffix",
        choices=("G", "H"),
        default="H",
        help="NHANES release suffix: G=2011-2012; H=2013-2014.",
    )
    return parser.parse_args()


def read_xpt(raw_dir: Path, name: str, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_sas(raw_dir / name, format="xport", encoding="latin-1")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing expected columns: {missing}")
    return frame.loc[:, columns].copy()


def valid_numeric(series: pd.Series, maximum: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if maximum is not None:
        values = values.mask(values > maximum)
    return values


def coded(series: pd.Series, allowed: tuple[int, ...]) -> pd.Series:
    values = valid_numeric(series)
    return values.where(values.isin(allowed))


def yes_no(series: pd.Series) -> pd.Series:
    values = coded(series, (1, 2))
    return values.map({1.0: 1.0, 2.0: 0.0})


def merge_one_to_one(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    if other["SEQN"].duplicated().any():
        raise ValueError("A one-row-per-participant source contains duplicate SEQN values.")
    return base.merge(other, on="SEQN", how="left", validate="one_to_one")


def build_labels(raw_dir: Path) -> pd.DataFrame:
    depression = read_xpt(raw_dir, cycle_file("DPQ"), ["SEQN", *PHQ_ITEMS])
    items = depression[PHQ_ITEMS].apply(valid_numeric, maximum=3)
    complete = items.notna().all(axis=1)
    total = items.sum(axis=1).where(complete)
    return pd.DataFrame(
        {
            "SEQN": depression["SEQN"].astype(np.int64),
            "phq9_total": total,
            "depression": (total >= 10).where(complete).astype("Int64"),
            "phq9_complete": complete.astype(np.int8),
        }
    )


def build_demographics(raw_dir: Path) -> pd.DataFrame:
    columns = [
        "SEQN",
        "RIAGENDR",
        "RIDAGEYR",
        "RIDRETH3",
        "DMDEDUC2",
        "DMDMARTL",
        "INDFMPIR",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ]
    frame = read_xpt(raw_dir, cycle_file("DEMO"), columns)
    renamed = frame.rename(
        columns={
            "RIAGENDR": "sex",
            "RIDAGEYR": "age_years",
            "RIDRETH3": "race_ethnicity",
            "DMDEDUC2": "education",
            "DMDMARTL": "marital_status",
            "INDFMPIR": "income_poverty_ratio",
            "WTMEC2YR": "survey_weight",
            "SDMVPSU": "survey_psu",
            "SDMVSTRA": "survey_stratum",
        }
    )
    renamed["SEQN"] = renamed["SEQN"].astype(np.int64)
    renamed["sex"] = coded(renamed["sex"], (1, 2))
    renamed["race_ethnicity"] = coded(
        renamed["race_ethnicity"], (1, 2, 3, 4, 6, 7)
    )
    renamed["education"] = coded(renamed["education"], (1, 2, 3, 4, 5))
    renamed["marital_status"] = coded(
        renamed["marital_status"], (1, 2, 3, 4, 5, 6)
    )
    renamed["age_years"] = valid_numeric(renamed["age_years"], maximum=120)
    renamed["income_poverty_ratio"] = valid_numeric(
        renamed["income_poverty_ratio"], maximum=5
    )
    return renamed


def build_examination(raw_dir: Path) -> pd.DataFrame:
    body = read_xpt(raw_dir, cycle_file("BMX"), ["SEQN", "BMXBMI", "BMXWAIST"])
    body = body.rename(columns={"BMXBMI": "bmi", "BMXWAIST": "waist_cm"})
    body["bmi"] = valid_numeric(body["bmi"], maximum=100)
    body["waist_cm"] = valid_numeric(body["waist_cm"], maximum=250)

    pressure_columns = [
        "SEQN",
        "BPXPLS",
        "BPXSY1",
        "BPXSY2",
        "BPXSY3",
        "BPXSY4",
        "BPXDI1",
        "BPXDI2",
        "BPXDI3",
        "BPXDI4",
    ]
    pressure = read_xpt(raw_dir, cycle_file("BPX"), pressure_columns)
    systolic = pressure[[f"BPXSY{index}" for index in range(1, 5)]].apply(
        valid_numeric, maximum=300
    )
    diastolic = pressure[[f"BPXDI{index}" for index in range(1, 5)]].apply(
        valid_numeric, maximum=200
    )
    pressure_out = pd.DataFrame(
        {
            "SEQN": pressure["SEQN"],
            "pulse_bpm": valid_numeric(pressure["BPXPLS"], maximum=250),
            "systolic_bp": systolic.mean(axis=1),
            "diastolic_bp": diastolic.mean(axis=1),
        }
    )
    return merge_one_to_one(body, pressure_out)


def build_questionnaires(raw_dir: Path) -> pd.DataFrame:
    blood_pressure = read_xpt(
        raw_dir,
        cycle_file("BPQ"),
        ["SEQN", "BPQ020", "BPQ040A", "BPQ080"],
    ).rename(
        columns={
            "BPQ020": "hypertension_diagnosed",
            "BPQ040A": "antihypertensive_use",
            "BPQ080": "high_cholesterol_diagnosed",
        }
    )
    for name in blood_pressure.columns.drop("SEQN"):
        blood_pressure[name] = yes_no(blood_pressure[name])

    diabetes = read_xpt(raw_dir, cycle_file("DIQ"), ["SEQN", "DIQ010", "DIQ160"])
    diabetes = diabetes.rename(
        columns={"DIQ010": "diabetes_diagnosed", "DIQ160": "prediabetes_diagnosed"}
    )
    for name in diabetes.columns.drop("SEQN"):
        diabetes[name] = yes_no(diabetes[name])

    health = read_xpt(raw_dir, cycle_file("HSQ"), ["SEQN", "HSD010"])
    health = health.rename(columns={"HSD010": "self_rated_health"})
    health["self_rated_health"] = valid_numeric(health["self_rated_health"], maximum=5)

    smoking = read_xpt(raw_dir, cycle_file("SMQ"), ["SEQN", "SMQ020", "SMQ040"])
    smoked_100 = coded(smoking["SMQ020"], (1, 2))
    current_status = coded(smoking["SMQ040"], (1, 2, 3))
    smoking["smoking_status"] = np.select(
        [smoked_100.eq(2), smoked_100.eq(1) & current_status.eq(3), current_status.isin([1, 2])],
        [0.0, 1.0, 2.0],
        default=np.nan,
    )
    smoking = smoking[["SEQN", "smoking_status"]]

    alcohol = read_xpt(raw_dir, cycle_file("ALQ"), ["SEQN", "ALQ101", "ALQ130"])
    alcohol = alcohol.rename(
        columns={"ALQ101": "alcohol_12plus_year", "ALQ130": "drinks_per_day"}
    )
    alcohol["alcohol_12plus_year"] = yes_no(alcohol["alcohol_12plus_year"])
    alcohol["drinks_per_day"] = valid_numeric(alcohol["drinks_per_day"], maximum=30)

    output = blood_pressure
    for frame in (diabetes, health, smoking, alcohol):
        output = merge_one_to_one(output, frame)
    return output


def build_chronic_conditions(raw_dir: Path) -> pd.DataFrame:
    frame = pd.read_sas(
        raw_dir / cycle_file("MCQ"),
        format="xport",
        encoding="latin-1",
    )
    available_items = [name for name in CHRONIC_ITEMS if name in frame.columns]
    if not available_items:
        raise ValueError(f"{cycle_file('MCQ')} has no expected chronic-condition items.")
    conditions = pd.DataFrame(
        {name: yes_no(frame[name]) for name in available_items}
    )
    observed = conditions.notna().sum(axis=1)
    count = conditions.fillna(0).sum(axis=1).where(observed > 0)
    return pd.DataFrame({"SEQN": frame["SEQN"], "chronic_condition_count": count})


def activity_minutes(frame: pd.DataFrame, days: str, minutes: str, met: float) -> pd.Series:
    days_per_week = valid_numeric(frame[days], maximum=7)
    minutes_per_day = valid_numeric(frame[minutes], maximum=1_440)
    return days_per_week * minutes_per_day * met


def build_extended_features(raw_dir: Path) -> pd.DataFrame:
    sleep = read_xpt(
        raw_dir,
        cycle_file("SLQ"),
        ["SEQN", "SLD010H", "SLQ050", "SLQ060"],
    )
    sleep = sleep.rename(
        columns={
            "SLD010H": "sleep_hours_weekday",
            "SLQ050": "trouble_sleeping_told_doctor",
            "SLQ060": "sleep_disorder_diagnosed",
        }
    )
    sleep["sleep_hours_weekday"] = valid_numeric(
        sleep["sleep_hours_weekday"], maximum=24
    )
    sleep["trouble_sleeping_told_doctor"] = yes_no(
        sleep["trouble_sleeping_told_doctor"]
    )
    sleep["sleep_disorder_diagnosed"] = yes_no(sleep["sleep_disorder_diagnosed"])

    activity_columns = [
        "SEQN",
        "PAQ610",
        "PAD615",
        "PAQ625",
        "PAD630",
        "PAQ640",
        "PAD645",
        "PAQ655",
        "PAD660",
        "PAQ670",
        "PAD675",
        "PAD680",
    ]
    activity = read_xpt(raw_dir, cycle_file("PAQ"), activity_columns)
    activity["self_report_met_minutes_week"] = pd.concat(
        [
            activity_minutes(activity, "PAQ610", "PAD615", 8.0),
            activity_minutes(activity, "PAQ625", "PAD630", 4.0),
            activity_minutes(activity, "PAQ640", "PAD645", 4.0),
            activity_minutes(activity, "PAQ655", "PAD660", 8.0),
            activity_minutes(activity, "PAQ670", "PAD675", 4.0),
        ],
        axis=1,
    ).sum(axis=1, min_count=1)
    activity["sedentary_minutes_day"] = valid_numeric(activity["PAD680"], maximum=1_440)
    activity = activity[["SEQN", "self_report_met_minutes_week", "sedentary_minutes_day"]]
    return merge_one_to_one(sleep, activity)


def build_medications(raw_dir: Path) -> pd.DataFrame:
    frame = read_xpt(
        raw_dir,
        cycle_file("RXQ_RX"),
        ["SEQN", "RXDUSE", "RXDDRUG", "RXDCOUNT"],
    )
    drug_names = frame["RXDDRUG"].fillna("").astype(str).str.lower()
    frame["antidepressant_use"] = drug_names.str.contains(
        ANTIDEPRESSANT_TERMS, regex=True
    ).astype(np.int8)
    frame["prescription_count"] = valid_numeric(frame["RXDCOUNT"], maximum=50)
    output = (
        frame.groupby("SEQN", as_index=False)
        .agg(
            prescription_count=("prescription_count", "max"),
            antidepressant_use=("antidepressant_use", "max"),
            prescription_use=("RXDUSE", "max"),
        )
        .reset_index(drop=True)
    )
    no_prescription = output["prescription_use"].eq(2)
    output.loc[no_prescription, "prescription_count"] = 0.0
    output.loc[no_prescription, "antidepressant_use"] = 0.0
    return output.drop(columns="prescription_use")


def feature_dictionary() -> pd.DataFrame:
    suffix = CYCLE_SUFFIX
    rows = [
        ("sex", f"DEMO_{suffix}", "core", "categorical", "Sex"),
        ("age_years", f"DEMO_{suffix}", "core", "numeric", "Age in years"),
        ("race_ethnicity", f"DEMO_{suffix}", "core", "categorical", "Race and ethnicity"),
        ("education", f"DEMO_{suffix}", "core", "categorical", "Education level"),
        ("marital_status", f"DEMO_{suffix}", "core", "categorical", "Marital status"),
        ("income_poverty_ratio", f"DEMO_{suffix}", "core", "numeric", "Income-to-poverty ratio"),
        ("bmi", f"BMX_{suffix}", "core", "numeric", "Body mass index"),
        ("waist_cm", f"BMX_{suffix}", "core", "numeric", "Waist circumference"),
        ("pulse_bpm", f"BPX_{suffix}", "core", "numeric", "Resting pulse"),
        ("systolic_bp", f"BPX_{suffix}", "core", "numeric", "Mean systolic blood pressure"),
        ("diastolic_bp", f"BPX_{suffix}", "core", "numeric", "Mean diastolic blood pressure"),
        ("hypertension_diagnosed", f"BPQ_{suffix}", "core", "binary", "Diagnosed hypertension"),
        ("antihypertensive_use", f"BPQ_{suffix}", "core", "binary", "Antihypertensive use"),
        ("high_cholesterol_diagnosed", f"BPQ_{suffix}", "core", "binary", "Diagnosed high cholesterol"),
        ("diabetes_diagnosed", f"DIQ_{suffix}", "core", "binary", "Diagnosed diabetes"),
        ("prediabetes_diagnosed", f"DIQ_{suffix}", "core", "binary", "Diagnosed prediabetes"),
        ("self_rated_health", f"HSQ_{suffix}", "core", "categorical", "General health rating"),
        ("smoking_status", f"SMQ_{suffix}", "core", "categorical", "Never, former, or current smoking"),
        ("alcohol_12plus_year", f"ALQ_{suffix}", "core", "binary", "At least 12 drinks in one year"),
        ("drinks_per_day", f"ALQ_{suffix}", "core", "numeric", "Average drinks on drinking days"),
        ("chronic_condition_count", f"MCQ_{suffix}", "core", "numeric", "Count of conditions"),
        ("sleep_hours_weekday", f"SLQ_{suffix}", "extended", "numeric", "Weekday sleep duration"),
        ("trouble_sleeping_told_doctor", f"SLQ_{suffix}", "extended", "binary", "Sleep trouble"),
        ("sleep_disorder_diagnosed", f"SLQ_{suffix}", "extended", "binary", "Sleep disorder"),
        ("self_report_met_minutes_week", f"PAQ_{suffix}", "extended", "numeric", "MET-minutes/week"),
        ("sedentary_minutes_day", f"PAQ_{suffix}", "extended", "numeric", "Sedentary minutes/day"),
        ("prescription_count", f"RXQ_RX_{suffix}", "sensitivity", "numeric", "Prescription count"),
        ("antidepressant_use", f"RXQ_RX_{suffix}", "sensitivity", "binary", "Antidepressant use"),
    ]
    return pd.DataFrame(rows, columns=["feature", "source", "role", "kind", "description"])


def main() -> None:
    global CYCLE_SUFFIX
    args = parse_args()
    CYCLE_SUFFIX = args.cycle_suffix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = build_labels(args.raw_dir)
    covariates = build_demographics(args.raw_dir)
    for frame in (
        build_examination(args.raw_dir),
        build_questionnaires(args.raw_dir),
        build_chronic_conditions(args.raw_dir),
        build_extended_features(args.raw_dir),
        build_medications(args.raw_dir),
    ):
        covariates = merge_one_to_one(covariates, frame)

    dictionary = feature_dictionary()
    expected = set(dictionary["feature"])
    observed = set(covariates.columns)
    if not expected.issubset(observed):
        raise RuntimeError(f"Missing engineered features: {sorted(expected - observed)}")

    labels.to_csv(args.output_dir / "labels_from_dpq.csv", index=False)
    covariates.to_csv(args.output_dir / "covariates.csv", index=False)
    dictionary.to_csv(args.output_dir / "feature_dictionary.csv", index=False)
    analysis_covariates = labels.loc[labels["phq9_complete"].eq(1), ["SEQN"]].merge(
        covariates,
        on="SEQN",
        how="left",
        validate="one_to_one",
    )
    missingness = analysis_covariates[list(dictionary["feature"])].isna().mean().sort_values()
    summary = {
        "participants_with_dpq": int(len(labels)),
        "complete_phq9": int(labels["phq9_complete"].sum()),
        "depression_positive": int(labels["depression"].fillna(0).sum()),
        "covariate_participants": int(len(covariates)),
        "engineered_features": int(len(dictionary)),
        "missing_fraction": {name: float(value) for name, value in missingness.items()},
        "leakage_control": "DPQ items are used only to derive labels and never enter covariates.csv.",
    }
    with (args.output_dir / "preparation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
