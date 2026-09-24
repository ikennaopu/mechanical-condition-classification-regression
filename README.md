# Mechanical Condition Classification and Regression

This project uses the `mechanical.csv` dataset to build neural-network models for predicting mechanical equipment condition.

## Models

### Classification

The classification model predicts one of three condition classes: `0`, `1`, or `2`.

- Output layer: 3-node softmax
- Loss: `sparse_categorical_crossentropy`
- Metric: Accuracy

Because `condition_class` represents categories, classification is the recommended approach.

### Regression

The regression model treats the condition labels as numeric values for comparison.

- Output layer: 1-node linear output
- Loss: Mean squared error (`mse`)
- Metrics: Mean absolute error and RMSE

This regression model is provided for demonstration only. The values `0`, `1`, and `2` are class labels and do not necessarily represent meaningful continuous measurements.

## Features

The models use:

- Rotational speed
- Load
- Bearing temperature
- Lubricant level
- Vibration

The features are standardized before training.

## Visualizations

The script generates plots showing:

- Classification training and validation loss
- Classification training and validation accuracy
- Regression training and validation loss
- Regression training and validation mean absolute error

## Files

- `meche-data.py` - Python implementation
- `meche-data.ipynb` - Jupyter Notebook version
- `mechanical.csv` - Original dataset
- `X_train.csv`, `X_test.csv` - Feature datasets
- `y_train.csv`, `y_test.csv` - Target datasets

## Requirements

- Python 3.12.8
- TensorFlow
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook

Install the dependencies with:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib jupyter
```

## Running the Project

Run the Python script:

```bash
python meche-data.py
```

Or open and run all cells in:

meche-data.ipynb

## Purpose

This project demonstrates the difference between classification and regression neural networks using the same mechanical dataset.