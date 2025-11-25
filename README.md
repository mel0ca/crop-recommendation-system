```markdown
# Crop Recommendation System

This project implements a machine learning-based crop recommendation system. The system helps farmers identify the most suitable crops to cultivate in their land based on various environmental parameters. By analyzing nutrient levels (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH value, and rainfall, the model predicts the optimal crop, enhancing agricultural productivity and sustainability.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to develop an intelligent system that can recommend the best crop to grow in a specific agricultural field. This is achieved by training various classification models on a diverse dataset of environmental factors and their corresponding crop types. The system aims to assist farmers in making informed decisions, leading to better yield and efficient resource utilization.

## Dataset

The dataset used for this project is `Crop_recommendation.csv`. It contains the following features:

- `N`: Nitrogen content in the soil
- `P`: Phosphorus content in the soil
- `K`: Potassium content in the soil
- `temperature`: Ambient temperature (in Celsius)
- `humidity`: Relative humidity (in %)
- `ph`: pH value of the soil
- `rainfall`: Rainfall (in mm)
- `label`: The recommended crop type (target variable)

**Data Preprocessing Steps:**
1.  **Loading Data**: The dataset was loaded using Pandas.
2.  **Exploratory Data Analysis (EDA)**: Basic information about the dataset, descriptive statistics, null values, and duplicated rows were checked. The distribution of crop labels was also visualized.
3.  **Feature and Target Separation**: The `label` column was separated as the target variable (`y`), and the remaining columns as features (`x`).
4.  **Label Encoding**: The categorical `label` column was converted into numerical format using `LabelEncoder`.
5.  **Train-Test Split**: The data was split into training and testing sets (80% training, 20% testing) to evaluate model performance.
6.  **Feature Scaling**: Numerical features were scaled using `StandardScaler` to ensure that all features contribute equally to the model training.

## Methodology

This project employs a supervised machine learning approach to build the crop recommendation system. Several classification algorithms were trained and evaluated to find the best performing model. The general steps involved:

1.  **Data Collection and Preprocessing**: As described in the Dataset section.
2.  **Model Training**: Various classification models were trained on the preprocessed training data.
3.  **Model Evaluation**: The performance of each model was assessed using accuracy score and classification reports on the test set. A confusion matrix was also generated for the best performing model to visualize its predictions.

## Models Used

The following machine learning models were trained and evaluated:

-   **Logistic Regression**
-   **Gaussian Naive Bayes (GaussianNB)**
-   **Support Vector Machine (SVM)**
-   **Random Forest Classifier**
-   **Decision Tree Classifier**
-   **Extra Tree Classifier**
-   **Gradient Boosting Classifier**
-   **AdaBoost Classifier**
-   **Bagging Classifier**
-   **K-Nearest Neighbors (KNeighborsClassifier)**

## Results

The accuracy of each model on the test set was recorded:

-   **Logistic Regression**: 0.9636
-   **GaussianNB**: 0.9955
-   **SVM**: 0.9682
-   **RandomForest**: 0.9932
-   **Decisiontree**: 0.9886
-   **ExtraTree**: 0.8773
-   **Gradientboosting**: 0.9818
-   **Adaboost**: 0.1455
-   **Bagging**: 0.9886
-   **Kneighbors**: 0.9568

The **Gaussian Naive Bayes** model achieved the highest accuracy of **0.9955**, making it the best-performing model for this dataset. A confusion matrix for this model (or the last trained model, Kneighbors in the current output) was generated to visualize the prediction performance across different crop types.

## Installation

To run this project, you'll need Python and the following libraries. You can install them using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd crop-recommendation-system
    ```
2.  **Place the dataset:** Ensure the `Crop_recommendation.csv` file is in the same directory as your Python script or notebook.
3.  **Run the script/notebook:** Execute the Python script or Jupyter Notebook cells sequentially.

    The code will:
    -   Load and preprocess the data.
    -   Train and evaluate multiple machine learning models.
    -   Print the accuracy of each model.
    -   Display a confusion matrix for the last trained model.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
