# Breast Cancer Classification from RNA-Seq Data

---

## Dataset

- **Source:** [GDC Portal](https://portal.gdc.cancer.gov/)
- **Project:** TCGA-BRCA
- **Data type:** RNA-Seq Gene Expression Quantification (TSV)
- **Samples:** 1,231 total (1,118 tumor, 113 normal)
- **Imbalance ratio:** 9.9:1
- **Feature:** `tpm_unstranded` (TPM values)



## Pipeline

```
Data Collection (TCGA-BRCA · GDC portal · RNA-Seq TSV)
        ↓
Preprocessing (Protein-coding filter · log₂(TPM+1))
        ↓
Train / Test Split (Stratified 80/20)
        ↓
Feature Selection (Variance-based · top 2,000 genes · train only)
        ↓
Class Imbalance Handling (7 strategies)
        ↓
Cross-Validation (5-fold stratified · per strategy)
        ↓
Best Strategy Selection (Multi-criteria: AUC · F1 · stability · Recall)
        ↓
Final Model Training (Random Forest · SMOTEENN · full train set)
        ↓
Test Set Evaluation          Permutation Test
        ↓                           ↓
        └──────────────┬────────────┘
                       ↓
            Model Explainability
        (MDI · Permutation Importance · SHAP)
```

---

## Class Imbalance Handling Strategies

| Strategy | Description |
|----------|-------------|
| None (Baseline) | No balancing applied |
| Class Weighting | Inverse class frequency weights |
| Random Oversampling | Duplicate minority class samples |
| Random Undersampling | Remove majority class samples |
| SMOTE | Synthetic Minority Over-sampling Technique |
| SMOTEENN | SMOTE + Edited Nearest Neighbours cleaning |
| SMOTETomek | SMOTE + Tomek Links cleaning |

---

## Results Summary

### Best Strategy: SMOTEENN

| Metric | CV (mean ± std) | Test Set |
|--------|----------------|----------|
| AUC-ROC | 1.0000 ± 0.0000 | 0.9988 |
| F1 Score | 1.0000 ± 0.0000 | 0.9955 |
| Accuracy | 1.0000 ± 0.0000 | 0.9919 |
| Precision | 1.0000 ± 0.0000 | 1.0000 |
| Recall | 1.0000 ± 0.0000 | 0.9911 |

### Permutation Test
- Real AUC-ROC: **0.9988**
- Mean permuted AUC-ROC: **0.5034**
- p-value: **0.0000** (n=1,000 permutations)

### Consensus Biomarker Genes (MDI + Permutation + SHAP)
| Gene | Role |
|------|------|
| CD300LG | Immune regulation · TNBC prognostic biomarker |
| CAVIN2 | Metastasis suppressor · caveolae-associated protein |
| ITIH5 | Tumour suppressor · ECM modulator |

---

## Project Structure

```
breast-cancer-detection/
├── _raw_data/                          # Raw GDC data (not included in repo)
│   ├── samples/                        # RNA-Seq TSV files (one subfolder per file UUID)
│   └── gdc_sample_sheet.tsv            # GDC sample sheet
├── pipelines/
│   ├── run_data_loading.py             # Step 1: Load raw TSV files → feature matrix
│   ├── run_training.py                 # Step 2: Train/test split + feature selection + CV
│   ├── run_evaluation.py               # Step 3: Test set evaluation
│   ├── run_explainability.py           # Step 4: MDI + Permutation + SHAP
│   ├── run_comparison.py               # Step 5: Cross-strategy comparison plots
│   ├── run_permutation_test.py         # Step 6: Permutation significance test
│   ├── run_data_exploration.py         # Step 7: Exploratory data analysis plots
│   └── run_all.py                      # Run full pipeline end-to-end
├── src/
│   ├── config.py                       # All configuration parameters
│   ├── utils.py                        # Shared utilities
│   ├── data_loader.py                  # Load GDC TSV files + gene name mapping
│   ├── preprocessor.py                 # Low-expression filter + log transformation
│   ├── feature_selector.py             # Variance-based feature selection
│   ├── balancer.py                     # 7 class imbalance handling strategies
│   ├── model.py                        # Random Forest training + strategy selection
│   ├── evaluator.py                    # Evaluation metrics + confusion matrix
│   ├── explainer.py                    # MDI + Permutation + SHAP explainability
│   └── comparator.py                   # Cross-strategy comparison plots
├── results/
│   ├── metrics/                        # CSV/JSON results files
│   └── figures/                        # All generated figures
└── README.md
```

---

## Installation

```bash
git clone https://github.com/BabakBasharirad/breast-cancer-detection.git
cd breast-cancer-detection
pip install -r requirements.txt
```

### Requirements

```
python>=3.10
scikit-learn==1.7.2
imbalanced-learn==0.14.0
shap==0.51.0
numpy==2.3.5
pandas==2.3.3
matplotlib==3.10.6
seaborn==0.13.2
tqdm==4.67.1
```

---

## Data Preparation

The raw data can be obtained in two ways:

### Option 1 — Using the GDC Data Transfer Tool (Recommended)

A GDC manifest file (`_raw_data/gdc_manifest.txt`) and sample sheet (`_raw_data/gdc_sample_sheet.tsv`) are provided in this repository to allow exact reproduction of the dataset used in this study.

1. Install the [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)
2. Run the following command from the project root:

```bash
gdc-client download -m _raw_data/gdc_manifest.txt -d _raw_data/samples/
```

3. This will download all 1,231 TSV files into `_raw_data/samples/`, with one subfolder per file UUID.

### Option 2 — Manual Download from GDC Portal

1. Go to the [GDC Portal](https://portal.gdc.cancer.gov/)
2. Apply the query parameters listed in the Dataset section above
3. Download the files and place them in `_raw_data/samples/` (one subfolder per file UUID)

---

## Usage

### Run Full Pipeline

```bash
python pipelines/run_all.py
```

### Run Individual Steps

```bash
# Step 1: Load data
python pipelines/run_data_loading.py

# Step 2: Train models (all 7 strategies)
python pipelines/run_training.py

# Step 3: Evaluate on test set
python pipelines/run_evaluation.py

# Step 4: Model explainability
python pipelines/run_explainability.py

# Step 5: Cross-strategy comparison
python pipelines/run_comparison.py

# Step 6: Permutation test
python pipelines/run_permutation_test.py

# Step 7: Data exploration plots
python pipelines/run_data_exploration.py
```

---

## Configuration

All pipeline parameters are defined in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_TOP_FEATURES` | 2000 | Number of top genes to select |
| `MIN_TPM_THRESHOLD` | 0.1 | Minimum TPM for low-expression filter |
| `MIN_SAMPLE_FRACTION` | 0.2 | Minimum fraction of samples for expression |
| `TEST_SIZE` | 0.2 | Test set proportion |
| `CV_FOLDS` | 5 | Number of cross-validation folds |
| `N_ESTIMATORS` | 500 | Number of Random Forest trees |
| `RANDOM_STATE` | 42 | Random seed for reproducibility |
| `N_PERMUTATIONS` | 1000 | Number of permutation test iterations |
| `N_TOP_IMPORTANT_FEATURES` | 20 | Top genes to show in importance plots |

---

## Reproducibility

All results are fully reproducible using `RANDOM_STATE=42`. The following files are saved to `results/metrics/`:

- `full_results_all_strategies.csv` — CV and test metrics for all strategies
- `best_strategy.json` — selected best strategy and criteria
- `permutation_test.json` — permutation test results
- `gene_name_mapping.json` — Ensembl ID to gene name mapping
- `gene_count_summary.json` — gene counts at each filtering stage
- `data_summary.json` — dataset statistics

---



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
