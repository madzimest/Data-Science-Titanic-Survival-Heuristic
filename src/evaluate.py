"""
Evaluate heuristic accuracy on the Titanic dataset.
Usage: python src/evaluate.py  (from project root)
"""

import pandas as pd
from heuristics import complex_heuristic, custom_heuristic   # <-- corrected

def load_data(filepath):
    """Load Titanic data from CSV."""
    df = pd.read_csv(filepath)
    return df

def calculate_accuracy(predictions, true_labels):
    """Compute accuracy given predictions dict and true labels Series (indexed by PassengerId)."""
    correct = 0
    total = len(predictions)
    for pid, pred in predictions.items():
        if pid in true_labels.index and pred == true_labels.loc[pid]:
            correct += 1
    return correct / total

def main():
    # Path to dataset (adjust as needed; here relative to project root)
    
    data_path = "notebooks/data/titanic.csv"
    df = load_data(data_path)
    
    # True labels
    true_survived = df.set_index('PassengerId')['Survived']
    
    # Test complex heuristic
    preds_complex = complex_heuristic(df)
    acc_complex = calculate_accuracy(preds_complex, true_survived)
    print(f"Complex heuristic accuracy: {acc_complex:.2%}")
    
    # Test custom heuristic
    preds_custom = custom_heuristic(df)
    acc_custom = calculate_accuracy(preds_custom, true_survived)
    print(f"Custom heuristic accuracy: {acc_custom:.2%}")

if __name__ == "__main__":
    main()
