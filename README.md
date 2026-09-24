# 🧠 Machine Learning Algorithms from Scratch (NumPy)

Classic machine-learning algorithms **implemented from scratch with NumPy**, each validated with **unit tests** and compared against scikit-learn in notebooks. The aim was to understand the maths behind every `model.fit()`.

> Lab series for *Machine Learning* (B.Sc. Mathematical Engineering & AI, Universidad Pontificia Comillas ICAI, 2025/26). The course provided the function signatures, test suites and datasets. The implementations and analyses are mine.

✅ **41 unit tests passing**

---

## 📂 Contents

| # | Topic | Implemented from scratch |
|---|---|---|
| 01 | **k-Nearest Neighbours** | Minkowski distance; `fit` / `predict` / `predict_proba`; classification metrics (accuracy, precision, recall, specificity, F1); ROC curve, calibration curve and probability histograms; 2-D decision boundaries |
| 02 | **Linear regression** | Simple and multiple OLS via the normal equations; R², RMSE, MAE; comparison with scikit-learn; Anscombe's quartet |
| 03 | **Linear regression II** | **Gradient descent** vs least squares, one-hot encoding |
| 04 | **Logistic regression + regularisation** | Sigmoid, log-likelihood and gradient-descent training with **L1 (Lasso), L2 (Ridge) and Elastic Net** penalties; full classification report |
| 05 | **Cross-validation** | k-fold CV written by hand for model selection |
| 06 | **Linear Discriminant Analysis** | Within- and between-class scatter matrices, generalised eigenproblem (S_W⁻¹ S_B), projection and classification |
| 07 | **Decision trees & ensembles** | *(scikit-learn, not from scratch)* regression and classification trees, **cost-complexity pruning** (`ccp_alpha`), bagging and random forests on the Carseats and Hitters datasets |

Each folder follows the same layout:
```
0X-topic/
├── src/        # implementation
├── tests/      # pytest suite
├── notebook/   # experiments, plots and comparison with scikit-learn
└── data/       # when needed
```

## ▶️ How to run
```bash
pip install -r requirements.txt
cd 01-knn && python -m pytest     # run the tests of any lab (all 6 suites pass)
jupyter notebook                  # open the notebooks
```
Every notebook runs top to bottom. Lab 07 draws the trees with Graphviz, so install it first ([graphviz.org/download](https://graphviz.org/download/)).

## 🧰 Tech stack
Python · NumPy · pandas · Matplotlib · Seaborn · scikit-learn (for comparison only) · pytest

## 👤 Author
**Mateo Gómez-Acebo Girardet**
