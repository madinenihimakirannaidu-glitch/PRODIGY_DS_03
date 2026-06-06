import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_bank_classifier():
    # 1. LOAD DATASET (Bank marketing datasets typically use a semicolon separator)
    csv_file = "bank.csv" 
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Make sure it is placed in this exact folder!")
        return

    df = pd.read_csv(csv_file, sep=';')
    print("--- Dataset Loaded ---")
    print(f"Dataset Shape: {df.shape}\n")

    # 2. PREPROCESSING & ENCODING
    # The target variable is 'y' (yes/no). Let's convert it to binary (1/0)
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Isolate features (X) and target variable (y)
    X = df.drop(columns=['y'])
    y = df['y']
    
    # Convert categorical text columns into numeric dummy indicator columns
    X = pd.get_dummies(X, drop_first=True)

    # 3. TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")

    # 4. INITIALIZE AND TRAIN MODEL
    # Max depth is capped at 5 to ensure readable evaluation charts and prevent overfitting
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print("Model Training Complete!\n")

    # 5. EVALUATE MODEL
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"--- Model Accuracy: {accuracy * 100:.2f}% ---")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 6. VISUALIZATIONS
    # Visualization 1: Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title('Bank Marketing Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('bank_confusion_matrix.png', dpi=300)
    plt.show()

    # Visualization 2: Structure Diagram of the Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        model, 
        feature_names=X.columns, 
        class_names=['No Purchase', 'Purchase'], 
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    plt.title('Trained Decision Tree Logic Diagram (Depth=5)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bank_decision_tree_structure.png', dpi=300)
    plt.show()
    
    print("Pipeline finished successfully! Charts saved to your directory.")

if __name__ == "__main__":
    build_bank_classifier()
