# Titanic Survival Prediction 

This project uses a Machine Learning pipeline to predict passenger survival on the Titanic using the Titanic dataset. It includes a comparative analysis of model performance **with SMOTE** to handle class imbalance.

## Objective
Develop a classification model to predict whether a passenger survived the Titanic disaster.

## Dataset Source
Dataset used: [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/brendan45774/test-file)

## Technologies Used
- Python
- Pandas, Seaborn, Matplotlib
- Scikit-learn
- imbalanced-learn (SMOTE)
- Random Forest Classifier

## How to Run
1. Clone this repository
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Python script:
    ```bash
    python titanic_survival_prediction.py
    ```

## Features
- Missing value handling
- Encoding of categorical variables
- Feature scaling
- Class imbalance handled using **SMOTE**
- Model evaluation with confusion matrix and classification report

## Results
The script prints out a **classification report** and **confusion matrix** for model performance **after applying SMOTE**.

## Conclusion
Applying SMOTE helps balance the dataset and improve precision for the minority class (`Survived = 1`). This enhances the model’s capability to correctly identify survivors.

---

Feel free to fork, explore, or enhance this repository!
