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
| Repeated materials | Rows with identical features (same material measured several times) merged into one row, target = median | **8,401 rows → 2,374 materials (6,027 rows removed)** |
| Constant columns | Drop features with only one value | 0 found |
| Extreme values | Flag values with \|z\| > 4 | **3,440 values in 41 columns, flagged and kept** |
| **Final** | | **15,170 unique materials × 82 columns** |

Columns with the most extreme values: range_FusionHeat (251), wtd_range_Density (210),
wtd_std_FusionHeat (206), wtd_range_atomic_mass (202), std_FusionHeat (191).

### Decisions and why (for the Methods/Dataset text)
- **Duplicates removed:** an exact copy of a row adds no information but would make a
  cluster look bigger and a test look more confident than it is.
- **Repeated materials merged:** the features are computed from the chemical formula, so
  rows with identical features are the same material measured more than once (different
  papers report different Tc). The research question is about materials, so each material
  must count once: otherwise heavily studied materials pull k-means centroids and make
  statistical tests look more confident than they are (rows would not be independent).
  The **median** Tc is used because it is robust to one outlying measurement. Side effect:
  the maximum Tc drops from 185 K to 143 K, because the 185 K value was one of several
  measurements of a material whose median is lower.
- **Extreme values kept, not deleted:** they are real measurements, not entry errors.
  Removing them could delete materials with unusual compositions, which may be a family of
  their own. Their effect on k-means is reduced later by standardisation and PCA (3.3).
- **|z| > 4 threshold** was chosen before looking at the results, not tuned.
- **Nothing is removed silently:** each step logs what it removed
  (`results/logs/cleaning_log.csv`), and a test checks rows removed = rows logged.

### Tests (`tests/test_clean.py`, 6 tests, all passing)
- Duplicates removed, all columns kept.
- Repeated materials merged into one row with the median target.
- Constant column dropped and recorded in the log.
- Rows removed equal rows logged (no silent data loss).
- Extreme values flagged but not removed.
- Missing target column raises an error.

### Limitations to mention
- Repeated measurements of the same material are merged (above), so each row is one
  material. Materials that are different but chemically very close (e.g. small doping
  changes) remain separate rows and may still be related, so independence is improved but
  not perfect. The Kaggle mirror has no chemical formulas (`unique_m.csv` in the UCI
  release), so families of related formulas could not be checked.

### Figures/tables for the paper
- No figures from this stage. The cleaning table above can go into the Dataset section.

---

## 3.2 eda — DONE

**Feeds paper sections:** Dataset, Methods (justification of each method choice).

### What it is for
Look at the shape of the data before choosing methods. Each figure supports one decision
later in the pipeline: which statistical test (3.6), whether to standardise (3.3), and
whether PCA is justified (3.3/K2).

### How to reproduce
```
python -m src.eda      # reads results/data/clean.csv -> results/eda/
```
Outputs are gitignored; they regenerate from the cleaned data.

### Outputs (`results/eda/`)
| File | What it shows |
|---|---|
| `target_distribution.pdf` | Histogram of critical temperature (linear + log), QQ plot vs normal |
| `feature_scales.pdf` | Standard deviation of all 81 features, log axis |
| `feature_distributions.pdf` | Histograms of the 8 `mean_*` features (one per property) |
| `correlation_matrix.pdf` | 81×81 Pearson correlation heatmap, grouped by property |
| `eda_summary.csv` | Numbers quoted below |
| `feature_describe.csv` | describe() table of every feature |

### Results and the decision each one supports

| Finding | Numbers | Decision |
|---|---|---|
| Target is skewed and **bimodal**: a large peak at low Tc (< 10 K) and a second peak near 80–90 K. QQ plot is far from the line. | mean 33.6 K, median 19.0 K, skew 0.88, range 0.0003–143 K; 37% of materials below 10 K, 17% at 70–100 K | Normality fails, so use **rank-based tests (Kruskal-Wallis)** or a permutation test in 3.6, not ANOVA. |
| Feature scales differ by orders of magnitude. | std from 0.28 (wtd_entropy_ElectronAffinity) to 4,097 (range_Density), ratio ≈ 14,500× | K-means uses Euclidean distance, so without scaling a few large-unit features would dominate. **Standardise all features** in 3.3. |
| Features are correlated, mostly in blocks within the same property (the 10 statistics of one property move together), plus some blocks across properties (e.g. atomic mass and density). | 5.1% of feature pairs have \|r\| > 0.8, 2.1% have \|r\| > 0.9; mean \|r\| 0.47 within a property vs 0.33 across | Redundant features get extra weight in distances, so **PCA (option K2)** is reasonable. The explained-variance curve in 3.3 confirms how many components are needed. |
| Individual features are skewed and several are multimodal (e.g. mean_Valence takes a few discrete levels, mean_FusionHeat and mean_Density have long right tails). | see `feature_distributions.pdf` | Supports keeping extreme values (3.1) and relying on standardisation + PCA rather than deleting points. |

### Notes for the writer
- The bimodal target is worth one sentence in the Results: it hints at two broad groups of
  superconductors (low-Tc and high-Tc, likely cuprates). Clusters are built **without** the
  target, so whether they line up with this split is a real question for 3.6, not a given.
- Suggested figures for the paper: `target_distribution.pdf` (justifies the test choice) and
  `correlation_matrix.pdf` (justifies PCA). The other plots can go in an appendix or be
  described in one line.
- Overall correlation is moderate (only ~5% of pairs above 0.8), so do not overstate
  redundancy; the strong part is within each property's block.

## 3.3 preprocess — TODO
