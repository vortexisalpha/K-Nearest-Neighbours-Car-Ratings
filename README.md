# K-Nearest Neighbours Car Ratings

Scikit-learn project that predicts car acceptability ratings using a K-Nearest Neighbours classifier. The model uses categorical vehicle attributes such as buying price, maintenance cost, number of doors, passenger capacity, boot size, and safety rating.

## What This Demonstrates

- Loading a tabular dataset with Pandas.
- Encoding categorical variables with `LabelEncoder`.
- Splitting data into training and test sets.
- Training a KNN classifier with scikit-learn.
- Evaluating accuracy and inspecting individual predictions.

## Features

The model predicts the `class` label from:

- `buying`
- `maint`
- `door`
- `persons`
- `lug_boot`
- `safety`

Output classes:

- `unacc`
- `acc`
- `good`
- `vgood`

## Run

```bash
pip install pandas numpy scikit-learn
python3 carpredict.py
```

The script prints the test accuracy and a list of predicted vs actual labels for the test split.

## Status

Learning project focused on classical ML preprocessing and KNN classification over categorical tabular data.
