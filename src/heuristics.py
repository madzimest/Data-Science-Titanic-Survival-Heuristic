"""
Titanic Survival Prediction Heuristics
---------------------------------------
This module contains heuristic-based prediction functions for the Titanic dataset.
"""

import pandas as pd
import numpy as np

def complex_heuristic(df):
    """
    Predict survival using the rule: female OR (first class AND under 18).
    Returns a dictionary {PassengerId: 0/1}
    """
    predictions = {}
    for _, passenger in df.iterrows():
        pid = passenger['PassengerId']
        is_female = (passenger['Sex'] == 'female')
        # Age may be NaN – treat as not under 18
        age = passenger['Age']
        is_under_18 = (not pd.isna(age)) and (age < 18)
        is_first_class = (passenger['Pclass'] == 1)
        
        if is_female or (is_first_class and is_under_18):
            predictions[pid] = 1
        else:
            predictions[pid] = 0
    return predictions


def custom_heuristic(df):
    """
    Custom rule: female OR (second‑class child under 11) OR (female embarked at S)
    """
    predictions = {}
    for _, passenger in df.iterrows():
        pid = passenger['PassengerId']
        age = passenger['Age']
        is_female = (passenger['Sex'] == 'female')
        is_child = (not pd.isna(age)) and (age < 11)
        is_second_class = (passenger['Pclass'] == 2)
        is_embarked_S = (passenger['Embarked'] == 'S')
        
        if is_female or (is_child and is_second_class) or (is_female and is_embarked_S):
            predictions[pid] = 1
        else:
            predictions[pid] = 0
    return predictions
