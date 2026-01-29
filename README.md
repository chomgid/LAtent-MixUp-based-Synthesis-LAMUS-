# LAtent Mixup-based Synthesis (LAMUS)

This repository contains the official Python package implementation of the paper **LAtent Mixup-based Synthesis (Tentative)**.

## Installation

You can install the package via `pip`. Run the following command in your terminal:

```bash
pip install DSTS
```

## Usage

Below is a simple example to help you get started with the `DSTS` package.

### Import the package

```python
from DSTS import dsts
```

### Load your dataset

```python
data = ...  # Your dataset here
```

### Construct a DSTS model

```python
mixup_model = dsts(sort=True, centering='double')
```

### Fit the model to your data

```python
mixup_model.fit(train_data)
```

### Generate synthetic data

```python
generated_data = mixup_model.generate(aug=1)
```

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
