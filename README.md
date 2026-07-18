```
 /$$    /$$ /$$$$$$$$  /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$$   /$$$$$$        /$$      /$$ /$$      
| $$   | $$| $$_____/ /$$__  $$|__  $$__//$$__  $$| $$__  $$ /$$__  $$      | $$$    /$$$| $$      
| $$   | $$| $$      | $$  \__/   | $$  | $$  \ $$| $$  \ $$| $$  \ $$      | $$$$  /$$$$| $$      
|  $$ / $$/| $$$$$   | $$         | $$  | $$  | $$| $$$$$$$/| $$$$$$$$      | $$ $$/$$ $$| $$      
 \  $$ $$/ | $$__/   | $$         | $$  | $$  | $$| $$__  $$| $$__  $$      | $$  $$$| $$| $$      
  \  $$$/  | $$      | $$    $$   | $$  | $$  | $$| $$  \ $$| $$  | $$      | $$\  $ | $$| $$      
   \  $/   | $$$$$$$$|  $$$$$$/   | $$  |  $$$$$$/| $$  | $$| $$  | $$      | $$ \/  | $$| $$$$$$$$
    \_/    |________/ \______/    |__/   \______/ |__/  |__/|__/  |__/      |__/     |__/|________/
```   

---

Vectora-ML is a lightweight machine learning library implementing classical machine learning algorithms entirely from scratch using NumPy. The project emphasizes algorithmic understanding, clean software architecture, reusable APIs, and comprehensive testing while maintaining minimal external dependencies. Implementations are independent of scikit-learn and are validated against trusted reference implementations where appropriate.

---

# Features

- Classical machine learning algorithms implemented from scratch
- Consistent estimator API (`fit`, `predict`, `transform`)
- Pure NumPy implementations
- Modular, object-oriented architecture
- Comprehensive test suite
- Minimal dependencies
- Easy to extend with new algorithms
- Explicit implementations focused on readability and correctness

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/vectora-ml.git
cd vectora-ml
```

Install in editable mode:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -r requirements.txt
```

---

# Quick Example

```python
from vectora_ml.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---


# Project Structure

```text
vectora_ml/
├── core/
│   ├── estimator.py
│   ├── exceptions.py
│   ├── metrics.py
│   └── validation.py
├── linear_model/
├── tree/
├── ensemble/
├── neighbors/
├── svm/
├── decomposition/
├── utils/
├── examples/
└── tests/
```

| Directory | Description |
|------------|-------------|
| `core/` | Base estimator classes, validation utilities, metrics, and library-wide exceptions. |
| `linear_model/` | Linear models such as Linear Regression and future linear algorithms. |
| `tree/` | Decision tree implementations for classification and regression. |
| `ensemble/` | Ensemble learning algorithms including Random Forest, AdaBoost, and Gradient Boosting. |
| `neighbors/` | Instance-based learning algorithms such as K-Nearest Neighbors. |
| `svm/` | Support Vector Machine implementations. |
| `decomposition/` | Dimensionality reduction algorithms including PCA. |
| `utils/` | Shared helper functions and internal utilities. |
| `examples/` | Example scripts demonstrating library usage. |
| `tests/` | Unit and regression tests validating algorithm correctness. |

---

# Testing

Vectora-ML uses **pytest** as its primary testing framework.

The testing strategy focuses on:

- Algorithm correctness
- Numerical consistency
- Input validation
- Exception handling
- Reproducibility

Where applicable, implementations are compared against trusted reference implementations to verify correctness while keeping production code independent of external machine learning libraries.

Run the complete test suite with:

```bash
pytest
```

---

# Development Principles

The library follows a consistent engineering approach across all modules.

- Uniform estimator API
- Standard `fit` / `predict` interface
- Comprehensive input validation
- Clear exception hierarchy
- Modular components with well-defined responsibilities
- Clean, maintainable implementations
- Type hints where appropriate
- Minimal coupling between modules

---


# Contributing

Contributions are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a feature branch.
3. Follow the existing coding style and project structure.
4. Add or update tests for any new functionality.
5. Submit a pull request with a clear description of the proposed changes.

Bug reports, feature requests, and documentation improvements are also encouraged through GitHub Issues.

---

# License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
