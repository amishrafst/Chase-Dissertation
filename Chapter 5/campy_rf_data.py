"""Build the modeling table (80 weather predictors and the label) from the workbook.

The 2019 copy ("Copy of Meteorological poultry data(w campy).xlsx", sheet
"Campylobacter") stores, for each of 10 weather variables, a block of 10
columns: SampleDay, OneDay, Samp.1, TwoDay, Samp.2 .. Samp.7, where Samp.k is
the mean of days 0..k (sampling day through k days before).

The published model (campy_rf_*.py) uses 80 predictors per variable block:
OneDay, TwoDay and Avg1.2 .. Avg1.7, where Avg1.k is the mean of days 1..k
(the day before sampling through k days before). Since
Samp.k = (day0 + day1 + .. + dayk) / (k + 1), it follows that
Avg1.k = ((k + 1) * Samp.k - day0) / k.

Label: the manuscript calls a sample positive if countable colonies grew on
Campy-Cefex, i.e. CampyCefexLog10CFU/mL > 0 (NOT the CampyCapetown column).

Input: data/Copy of Meteorological poultry data(w campy).xlsx, sheet "Campylobacter".
"""

from pathlib import Path

import pandas as pd

DEFAULT_XLSX = Path(__file__).resolve().parent / "data" / "Copy of Meteorological poultry data(w campy).xlsx"

META = ["SampleID", "Farm", "AvgNumBirds", "SampleType",
        "CampyCefexLog10CFU/mL", "CampySpeciesBAX", "CampyCef", "SamplingDate"]


def avg_from_day1(samp_k, sample_day, k):
    """Mean of days 1..k, given Samp.k (mean of days 0..k) and the day-0 value."""
    return ((k + 1) * samp_k - sample_day) / k


def cefex_label(log10_cfu):
    """'+' if any countable Campy-Cefex colonies, else '-'. None if missing."""
    if log10_cfu is None or pd.isna(log10_cfu):
        return None
    return "+" if float(log10_cfu) > 0 else "-"


def build_model_frame(raw):
    """Return a frame shaped like the scripts' input: 8 meta columns + 80 predictors.

    Column positions 0,1,2,4,5,7 are the ones campy_rf_*.py drop, so position 3
    is SampleType and position 6 is the CampyCef label, as in the original file.
    """
    out = pd.DataFrame({
        "SampleID": raw["SampleID"],
        "Farm": raw["Farm"],
        "AvgNumBirds": raw["AvgNumBirds"],
        "SampleType": raw["SampleType"],
        "CampyCefexLog10CFU/mL": raw["CampyCefexLog10CFU/mL"],
        "CampySpeciesBAX": raw["CampySpeciesBAX"],
        "CampyCef": raw["CampyCefexLog10CFU/mL"].map(cefex_label),
        "SamplingDate": raw["SamplingDate"],
    })
    weather = list(raw.columns[8:108])
    for b in range(0, 100, 10):
        block = weather[b:b + 10]
        day0, one, _samp1, two = block[0], block[1], block[2], block[3]
        samp = {2: block[4], 3: block[5], 4: block[6], 5: block[7], 6: block[8], 7: block[9]}
        stem = day0.replace("SampleDay", "")
        out[f"{stem}OneDayBefore"] = pd.to_numeric(raw[one], errors="coerce")
        out[f"{stem}TwoDayBefore"] = pd.to_numeric(raw[two], errors="coerce")
        d0 = pd.to_numeric(raw[day0], errors="coerce")
        avg_stem = samp[2].rsplit("Samp.", 1)[0]
        for k, col in samp.items():
            out[f"{avg_stem}1.{k}"] = avg_from_day1(pd.to_numeric(raw[col], errors="coerce"), d0, k)
    return out


def as_published(frame):
    """Row handling that reproduces the published inputs (774 feces, 772 soil).

    - Weather is recorded per farm and date, so a blank weather cell is filled
      from another sample taken on the same farm and date (B2-20, HE2-10 soil).
    - Feces H1-1 has a blank CampyCapetown entry and is absent from the final
      file (774 feces in the manuscript); drop it.
    Verified 2026-10-12: with these rules Chase's pipeline reproduces the Table 3
    test-set split exactly for both models and the feces confusion matrix exactly.
    """
    predictors = list(frame.columns[8:])
    out = frame.copy()
    out[predictors] = out.groupby(["Farm", "SamplingDate"])[predictors].transform(
        lambda s: s.fillna(s.dropna().iloc[0]) if s.notna().any() else s)
    out = out[~((out.SampleID == "H1-1") & (out.SampleType == "Feces"))]
    return out.dropna(subset=predictors + ["CampyCef"]).reset_index(drop=True)


def load(xlsx=DEFAULT_XLSX):
    raw = pd.read_excel(xlsx, sheet_name="Campylobacter")
    return build_model_frame(raw)
