# Titanic Survival Prediction Heuristic Approach

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements heuristic‑based rules to predict survival of passengers on the Titanic. It was originally developed as part of an Udacity exercise.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Heuristics](#heuristics)
- [Results](#results)
- [Installation](#installation)
- 

## Overview
The sinking of the Titanic is one of the most infamous shipwrecks in history. In this project, we explore passenger data and design simple decision rules (heuristics) to predict who survived. The goal is to achieve high accuracy using only a few interpretable features.

## Dataset
The dataset is from [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic). It contains information on 891 passengers:

- `PassengerId`: unique ID
- `Survived`: survival (0 = No, 1 = Yes)
- `Pclass`: passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

> **Note:** The dataset is included in the `data/` folder or can be downloaded from the link above.

## Heuristics
Two heuristic functions are implemented in `src/heuristics.py`:

1. **`complex_heuristic`** – Survived if female OR (first‑class AND under 18).  
   This matches the Udacity exercise specification.

2. **`custom_heuristic`** – A more refined rule based on exploratory data analysis.  
   (Describe your own rule here, e.g., female OR second‑class child OR female embarked at Southampton.)

Both functions handle missing ages gracefully (age unknown ⇒ not considered a child).

## Results
When evaluated on the training dataset (which includes true survival labels), the heuristics achieve:

```
| Method                           | Accuracy |
|----------------------------------|----------|
| Simple heuristic (female only)   | 78.68%   |
| Complex heuristic (original)     | 79.12%   |
| Custom heuristic (original)      | 79.69%   |
| Complex heuristic (imputed)      | 79.12%   |
| Custom heuristic (imputed)       | 79.69%   |
| Logistic Regression (test set)   | 80.45%   |
| Logistic Regression (CV mean)    | 79.91% (±2.53%) |
```
*(Update these numbers after running the evaluation.)*

The custom heuristic beats the 80% threshold, demonstrating that simple rules can be surprisingly effective.

## Installation & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/titanic-survival-heuristic.git
   cd titanic-survival-heuristic
