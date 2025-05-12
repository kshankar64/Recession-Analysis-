# Credit Card Fraud Detection Using Machine Learning

This project implements and compares multiple machine learning models for detecting fraudulent credit card transactions. The workflow includes data preprocessing, handling class imbalance, outlier detection, model training, evaluation, and visualization.

---

## 📊 Problem Statement

Credit card fraud is a significant issue for financial institutions and customers. The goal is to accurately identify fraudulent transactions within a highly imbalanced dataset, minimizing false positives and false negatives.

---

## 🗂️ Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset][5]
- **Description:** Contains anonymized features of transactions made by European cardholders in September 2013. There are 284,807 transactions, with only 492 (0.172%) labeled as fraud.
- **Attributes:** 31 columns (28 PCA-transformed features, 'Amount', 'Time', and 'Class' where 'Class' is 1 for fraud, 0 for genuine)

---

## 🏗️ Project Structure

- **Data Loading & Preprocessing:**  
  - Drops the 'Time' column  
  - Splits data into features (`X`) and target (`y`)
  - Handles missing values and scales features using `RobustScaler`
- **Exploratory Data Analysis:**  
  - Visualizes class distribution  
  - Computes and plots correlation matrices  
  - Examines feature distributions and outliers
- **Handling Imbalance & Outliers:**  
  - Downsamples majority class (non-fraud) for balanced training  
  - Detects and removes outliers using IQR for selected features  
  - Creates three training sets: full, downsampled with outliers, downsampled without outliers
- **Model Training:**  
  - Models: K-Nearest Neighbors, Random Forest, Logistic Regression, Decision Tree, Support Vector Machine  
  - Trained with and without cross-validation on different datasets
- **Evaluation:**  
  - Computes accuracy, AUC, confusion matrix, TPR, FPR  
  - Plots ROC curves for all models and datasets
- **Visualization:**  
  - Boxplots for feature distributions  
  - ROC curves for model comparison

---

## 🛠️ How to Run

1. **Install Dependencies**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. **Download Dataset**
   - Download `creditcard.csv` from [Kaggle][5] and update the file path in the script.

3. **Run the Script**
   - Execute the Python notebook or script in Jupyter or any Python environment.

---

## 🧪 Models Compared

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Logistic Regression**
- **Decision Tree**
- **Support Vector Machine (SVM)**

Each model is trained and evaluated on:
- The full (imbalanced) dataset
- Downsampled dataset (with and without outliers)

---

## 📈 Evaluation Metrics

- **Accuracy**
- **AUC (Area Under ROC Curve)**
- **Confusion Matrix**
- **True Positive Rate (TPR), False Positive Rate (FPR)**
- **Number of misclassifications for each class**
- **Training time**

Results are summarized in dataframes and compared visually using ROC curves.

---

## 📋 Key Findings

- Class imbalance is a major challenge; downsampling and outlier removal are explored to improve model performance.
- Random Forest and SVM typically perform best for this problem.
- ROC curves and AUC provide a clear comparison of model effectiveness.

---

## 📌 Notes

- The code suppresses warnings for cleaner output.
- All data preprocessing, model training, and evaluation steps are fully reproducible.
- You can extend the code to try other resampling techniques (e.g., SMOTE) or algorithms.

---

## 📚 References

- [Kaggle: Credit Card Fraud Detection Dataset][5]
- [scikit-learn documentation](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 🤝 Contributing

Feel free to fork the repository, open issues, or submit pull requests to improve the project.

---

## 📝 License

This project is for educational and research purposes.

---[5]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

*This README was generated for the code provided in the attached notebook and is inspired by best practices seen in similar machine learning projects[2][4][6].*

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/27626270/c425ca49-31bf-4dc0-9363-6875f825790f/paste.txt
[2] https://github.com/shakiliitju/Credit-Card-Fraud-Detection-Using-Machine-Learning
[3] https://towardsdatascience.com/structuring-your-machine-learning-project-with-mlops-in-mind-41a8d65987c9/
[4] https://github.com/LaurentVeyssier/Credit-Card-fraud-detection-using-Machine-Learning
[5] https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
[6] https://www.youtube.com/watch?v=BRaAU2vAG_c
[7] https://github.com/sahidul-shaikh/credit-card-fraud-detection/blob/main/README.md
[8] https://mlflow.org/docs/latest/recipes/
[9] https://repos.ecosyste.ms/hosts/GitHub/repositories/marcellin-d%2FFraud-Detection-in-Online-Transactions/readme
[10] https://www.kaggle.com/getting-started/186583
[11] https://github.com/othneildrew/Best-README-Template
[12] https://community.cloudera.com/t5/Community-Articles/Credit-Fraud-Prevention-Demo-A-Guided-Tour/ta-p/246392
[13] https://www.kaggle.com/code/gauravduttakiit/credit-card-fraud-detection
[14] https://github.com/ml-tooling/ml-project-template/blob/develop/README.md
[15] https://www.scribd.com/document/813987270/GitHub-peggy1502-fraud-detection-handbook-Machine-Learning-for-Credit-Card-Fraud-Detection-Practical-Handbook
[16] https://www.reddit.com/r/opensource/comments/txl9zq/next_level_readme/
[17] https://discuss.codecademy.com/t/predicting-credit-card-fraud-project/691830
[18] https://www.youtube.com/watch?v=a_PuGq8BQHM
[19] https://repos.ecosyste.ms/hosts/GitHub/repositories/Projects-Developer%2FCredit-Card-Fraud-Detection-Using-Machine-Learning/readme
[20] https://people.cs.nott.ac.uk/blaramee/teaching/msc/thesis/kinnaird23thesis.pdf
[21] https://www.reddit.com/r/MachineLearning/comments/c0fsni/p_tutorial_to_build_a_complete_project_in_credit/

---
Answer from Perplexity: pplx.ai/share
