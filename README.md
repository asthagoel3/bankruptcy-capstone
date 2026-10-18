# Predicting Corporate Bankruptcy from Financial Ratios

**Astha Goel** | Boston University, OMDS Module C: Data Science Capstone

Weekly notebooks for Milestone One. Each notebook applies that week's modeling topic to two
bankruptcy datasets and compares the results across countries.

## The project

This capstone predicts corporate bankruptcy from published financial ratios, and tests whether the
distress signal behaves the same way in two very different economies.

| | Poland | Taiwan |
|---|---|---|
| Observations | 43,405 firm-years (2000 to 2013) | 6,819 firms (1999 to 2009) |
| Features | 64 financial ratios | 95 financial ratios |
| Bankrupt | 2,091 (4.8%) | 220 (3.2%) |
| Source | UCI, Tomczak (2016) | UCI, Liang et al. (2016) |

The two datasets are never merged. Different accounting standards, currencies, and ratio
definitions make a row-wise join meaningless. Instead the same pipeline runs separately on each,
and the comparison happens at the level of findings rather than values.

## Main finding

Poland and Taiwan have different *functional forms* of bankruptcy risk, not just different
coefficients. Taiwanese risk is close to additive in the financial ratios. Polish risk depends on
interactions between them.

Three independent lines of evidence point the same way:

1. **Interaction terms** improve Poland by roughly 0.09 ROC-AUC and leave Taiwan unchanged. Checked
   across three train/test splits: Poland gains 0.07 to 0.10 every time, while Taiwan stays within
   0.02 and flips sign, so Taiwan's movement is noise (Week 1).
2. **SVM kernel choice** reproduces the pattern using an unrelated method. Taiwan's best kernel is
   linear; Poland's is the flexible RBF at C = 10 (Week 5).
3. **Model families** agree. In Taiwan a logistic regression (0.954) matches or beats the random
   forest (0.952). In Poland the forest (0.926) beats the best linear model (0.815) by a wide
   margin (Weeks 4 and 6).

## Results by week

All figures are test-set ROC-AUC on a held-out 20% split.

| Method | Poland | Taiwan |
|---|---|---|
| Wk1 Logistic + interaction terms | 0.878 | 0.923 |
| Wk2 Lasso / ridge / elastic net | 0.778 | 0.951 |
| Wk3 PCR (40 components) | 0.801 | 0.946 |
| Wk3 PLSR (best) | 0.787 | 0.937 |
| Wk3 Forward selection (10 features) | 0.800 | 0.938 |
| Wk4 Logistic + QuantileTransformer | 0.815 | 0.954 |
| Wk5 SVM, linear kernel | 0.854 | 0.947 |
| Wk5 SVM, RBF kernel (C = 10) | 0.882 | 0.925 |
| Wk6 Single decision tree (best depth) | 0.822 | 0.850 |
| Wk6 Random forest | 0.926 | 0.952 |

## Notebooks

| Notebook | Topic | What it found |
|---|---|---|
| [Week 1](notebooks/OMDS-ModC-Week1-Goel-Astha.ipynb) | Polynomial terms, interactions, multicollinearity, VIF, categorical features | A Taiwanese feature pair sits at VIF around 10^15, which is perfect multicollinearity from the accounting identity. Polynomial degree 3 helps Poland and overfits Taiwan. |
| [Week 2](notebooks/OMDS-ModC-Week2-Goel-Astha.ipynb) | Lasso, ridge, elastic net | Both datasets contain features ranging to 10^10 that break linear models until winsorized. Lasso removes near-duplicate ratio families automatically. |
| [Week 3](notebooks/OMDS-ModC-Week3-Goel-Astha.ipynb) | Forward selection, PCR, PLSR | Compression never beats the full feature set. Ten selected features get within about 0.015 of the full model and stay interpretable. |
| [Week 4](notebooks/OMDS-ModC-Week4-Goel-Astha.ipynb) | Logistic regression, feature scaling | Scaler choice swings Poland by 0.21 ROC-AUC. RobustScaler fails to converge. `class_weight='balanced'` wrecks calibration without improving ranking. |
| [Week 5](notebooks/OMDS-ModC-Week5-Goel-Astha.ipynb) | SVMs, kernel trick, regularization | Taiwan prefers the linear kernel; Poland prefers the flexible RBF. Independent confirmation of the main finding. |
| [Week 6](notebooks/OMDS-ModC-Week6-Goel-Astha.ipynb) | Decision trees, random forests | Unrestricted trees reach train ROC-AUC of exactly 1.000 while test collapses. The panel-leakage concern in the Polish data turns out not to matter. |

## Running the notebooks

```bash
pip install -r requirements.txt
jupyter lab
```

The datasets are not committed to this repo. Download them from UCI and place them in `data/`:

- `data/Polish_dataset_capstone.csv`
- `data/Taiwan_data_capstone.csv`

Each notebook reads the two files by name from the working directory, so either run Jupyter from
`data/`, or edit the two `pd.read_csv` paths at the top of each notebook.

Two things to check before running:

- The Polish file's ratio columns should be named `A1` through `A64`. If yours uses `Attr1` through
  `Attr64`, change the `rename_map` line in the loading cell.
- The Taiwanese file's column headers carry a leading space on every column except `Bankrupt?`.
  The loading cell strips this with `.str.strip()`. Do not remove that line.

## Notes on method

- **Scaling** uses `QuantileTransformer` inside a `Pipeline`, chosen in Week 4 after comparing six
  scalers. Putting the scaler in the pipeline keeps it fit on training folds only.
- **Splits** are stratified so the rare bankrupt class appears at its true rate in every fold.
- **Metric** is ROC-AUC rather than accuracy. With roughly 96% of firms solvent, predicting
  "solvent" for everything would already score about 96% while being useless.
- **Train and test scores are reported together** throughout, so overfitting is visible rather than
  inferred.

## Known limitations

- Polish SVMs (Week 5) use an 8,000-row stratified subsample, since kernel SVMs scale poorly with
  sample size.
- The Polish data has a panel structure, with 10,503 firms averaging about four observations each.
  Week 6 tests this with a group-aware split keyed on firm identity. The group split scores
  **higher** than the random split (0.9392 against 0.9237), so firm-level repetition was not
  inflating the results.

## References

Liang, D., Lu, C.-C., Tsai, C.-F., & Shih, G.-A. (2016). Financial ratios and corporate governance
indicators in bankruptcy prediction: A comprehensive study. *European Journal of Operational
Research*, 252(2), 561-572.

Tomczak, S. (2016). Polish Companies Bankruptcy Data [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5F600

Taiwanese Bankruptcy Prediction [Dataset]. (2020). UCI Machine Learning Repository.
https://doi.org/10.24432/C5004D
