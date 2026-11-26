# raw CSV -> load_clean -> eda
#                     \-> preprocess -> cluster -> stability -> stats_tests -> figures_tables -> all
# Rules stay thin: each one calls functions in src/.

rule all:
    input: []
