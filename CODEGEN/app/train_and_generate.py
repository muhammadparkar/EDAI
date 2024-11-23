
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from jinja2 import Environment, FileSystemLoader
import os

def train_and_generate_template(data_path="updated_dataset.csv"):
    # Step 1: Load and preprocess the dataset
    data = pd.read_csv(data_path)

    # Handle missing values (dropping rows where all target columns are missing)
    data = data.dropna(subset=["Optimizer", "Validation_Technique", "Loss_Function"], how="all")

    # Encode categorical columns
    label_encoders = {}
    for col in ["Optimizer", "Validation_Technique", "Loss_Function"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Features and target preparation
    features = data.drop(["Optimizer", "Validation_Technique", "Loss_Function"], axis=1)

    # Define targets
    targets = {
        "Optimizer": data["Optimizer"],
        "Validation_Technique": data["Validation_Technique"],
        "Loss_Function": data["Loss_Function"]
    }

    # Train models for each target
    models = {}
    for target_name, target in targets.items():
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {target_name}: {accuracy * 100:.2f}%")
        models[target_name] = model

    # Save models and label encoders
    joblib.dump(models, "models.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

    # Generate template using Jinja2
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('templates/ml_code_template.jinja2')
    rendered_template = template.render(models=models, label_encoders=label_encoders)

    output_file = "generated_code.py"
    with open(output_file, "w") as f:
        f.write(rendered_template)

    print(f"Generated code has been saved to {output_file}.")
    return output_file

if __name__ == "__main__":
    train_and_generate_template()