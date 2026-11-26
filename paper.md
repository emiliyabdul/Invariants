# Notes for the paper writer

Information from each pipeline stage that the paper needs: what was done, why, results,
and where the files are. This is not the paper text.

---

## 3.1 load_clean — DONE

**Feeds paper sections:** Dataset, Limitations.

### What it is for
Read the raw data, check its quality, and remove or flag problems before any analysis.
Every later result depends on this step: k-means is sensitive to outliers, and duplicate
rows can inflate cluster sizes and the confidence of statistical tests.

### Data source
- UCI Superconductivity dataset, original paper: Hamidieh, K. (2018). *A data-driven
  statistical model for predicting the critical temperature of a superconductor.*
  Computational Materials Science, 154, 346–354.
- Downloaded from the Kaggle mirror `munumbutt/superconductor-dataset` (no login needed).
- Raw file: `data/raw/train.csv`, 21,263 rows × 82 columns.
- 81 numeric features + target `critical_temp` (critical temperature, K).
- Features: 8 elemental properties (atomic mass, first ionisation energy, atomic radius,
  density, electron affinity, fusion heat, thermal conductivity, valence) × 10 summary
  statistics (mean, weighted mean, geometric mean, weighted geometric mean, entropy,
  weighted entropy, range, weighted range, std, weighted std) + number of elements.

### How to reproduce
```
python src/download_data.py   # -> data/raw/train.csv
python src/load_clean.py      # -> results/data/clean.csv, results/logs/cleaning_log.csv
pytest tests/test_clean.py
```
Outputs are gitignored; they regenerate from the raw data.

### What was done and the results

| Step | Rule | Result |
|---|---|---|
| Load | Check target column exists and all columns are numeric | OK |
| Missing values | Drop rows with any missing value | 0 found |
| Duplicates | Drop exact duplicate rows | **66 rows removed** |
| Constant columns | Drop features with only one value | 0 found |
| Extreme values | Flag values with \|z\| > 4 | **4,698 values in 43 columns, flagged and kept** |
| **Final** | | **21,197 rows × 82 columns** |

Columns with the most extreme values: range_FusionHeat (327), wtd_range_Density (282),
wtd_range_atomic_mass (269), std_FusionHeat (269), wtd_mean_Density (252).

### Decisions and why (for the Methods/Dataset text)
- **Duplicates removed:** an exact copy of a row adds no information but would make a
  cluster look bigger and a test look more confident than it is.
- **Extreme values kept, not deleted:** they are real measurements, not entry errors.
  Removing them could delete materials with unusual compositions, which may be a family of
  their own. Their effect on k-means is reduced later by standardisation and PCA (3.3).
- **|z| > 4 threshold** was chosen before looking at the results, not tuned.
- **Nothing is removed silently:** each step logs what it removed
  (`results/logs/cleaning_log.csv`), and a test checks rows removed = rows logged.

### Tests (`tests/test_clean.py`, 5 tests, all passing)
- Duplicates removed, all columns kept.
- Constant column dropped and recorded in the log.
- Rows removed equal rows logged (no silent data loss).
- Extreme values flagged but not removed.
- Missing target column raises an error.

### Limitations to mention
- The same material (or very similar compositions) can appear several times with different
  critical temperatures. The Kaggle mirror has no chemical formulas (`unique_m.csv` in the
  UCI release), so only exact duplicate rows could be checked. Rows may not be fully
  independent, which weakens the independence assumption of the statistical tests.

### Figures/tables for the paper
- No figures from this stage. The cleaning table above can go into the Dataset section.

---

## 3.2 eda — TODO

## 3.3 preprocess — TODO
