# CI/CD for Machine Learning Development

## Event Information

🗓️ This repository was created for a talk at the Keras Community Day by GDG Bandung on September 16, 2023 at BINUS Bandung.

## Description

This repository demonstrates a complete machine learning workflow from development to deployment, leveraging GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD).

## Features

- Automated model training and evaluation
- Comparison of model metrics across different code branches
- Automated unit and integration tests
- Auto-commenting of evaluation metrics in pull requests by GitHub Actions

## Requirements

To install the project dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### For Training

```bash
python main.py --train
```

### For Testing

```bash
python main.py --test
```

### For Experimentation with New Code (Training)

```bash
python main.py --train --experiment
```

### For Experimentation with New Code (Testing)

```bash
python main.py --test --experiment
```

## GitHub Actions Output

The metrics and evaluation will be automatically posted as comments by GitHub Actions in the pull requests.
