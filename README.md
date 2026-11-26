# Invariants

Do superconducting materials fall into natural families by their features, are those families stable,
and do they differ in critical temperature?

## Setup

```bash
conda env create -f environment.yml
```

## Reproduce

```bash
snakemake --cores 1
```

## Layout

```
Snakefile          raw data -> every figure, one command
data/raw/train.csv raw data (not committed)
src/               load_clean, eda, preprocess, cluster, stats_tests, figures
tests/             pytest tests
results/           generated only; never edited by hand
```

## Contribution table

| Member | Contribution |
|--------|--------------|
|        |              |
