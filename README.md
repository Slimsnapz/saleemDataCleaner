# Saleem Data Cleaner

## Project Overview

The **Saleem Data Cleaner** is a comprehensive Python-based tool for data cleaning and preprocessing. It is designed to automate the most commonly needed tasks in data wrangling, ensuring datasets are ready for further analysis and modeling. The tool efficiently handles missing values, data type mismatches, outliers, feature transformations, and more, allowing data scientists and engineers to streamline their workflows.

## Functionality

The **Saleem Data Cleaner** provides a series of modular functions that users can either run independently or as part of a full data preprocessing pipeline. The key features include:

- **File Loading**: Load CSV or Excel files into a pandas DataFrame.
- **Data Shape and Information**: Quickly assess the number of rows and columns, and display metadata such as data types, missing values, and memory usage.
- **Categorical & Numerical Feature Splitting**: Automatically separate numerical and categorical columns.
- **Data Type Correction**: Detects and corrects data type mismatches, such as numerical values stored as strings.
- **Handling Missing Values**: Detect, treat, or remove missing values based on the user's threshold settings.
- **Duplicate Detection and Removal**: Find and remove duplicate rows from the dataset.
- **Label Encoding**: Handle both nominal and ordinal categorical features using One-Hot Encoding and Label Encoding.
- **Outlier Detection**: Use Interquartile Range (IQR) to detect and address outliers.
- **Normalization and Standardization**: Apply normalization (L2) and scaling (MinMaxScaler, StandardScaler) to numerical features.
- **Skewness Analysis**: Identify skewed distributions in numerical columns to determine if transformation is needed.
- **Feature Selection**: Perform feature selection using methods like Univariate Feature Selection (UFS), Recursive Feature Elimination (RFE), and Principal Component Analysis (PCA).

## How It Can Be Applied

This tool is perfect for data preprocessing tasks in machine learning workflows. It can be applied to both large and small datasets, offering flexibility to:

1. Clean data from various sources such as CSV and Excel.
2. Automatically identify and fix common data quality issues.
3. Prepare the dataset by encoding categorical variables and scaling numerical features.
4. Optimize datasets by selecting only the most important features for model building.
5. Perform end-to-end data wrangling or isolate specific functions like feature selection or handling missing values.

## Results Achieved

With the **Saleem Data Cleaner**, users can expect:

- **Cleaner datasets**: Automatically detect and handle common issues such as missing values, duplicates, and incorrect data types.
- **Well-processed numerical data**: Numerical features can be scaled or normalized to improve the performance of machine learning models.
- **Effective feature engineering**: Categorical and numerical features are prepared using proper encoding techniques, and redundant or less important features are removed using feature selection methods.
- **Faster workflows**: Preprocessing pipelines can be automated, saving time on repetitive tasks and reducing errors in data preparation.

## How to Use

### Installation

Ensure you have the necessary Python libraries installed by running:

```bash
pip install pandas numpy scikit-learn
```
Import the function
```bash
import saleemDataCleaner as sd
```
After installing the neccessary libraries and importing the module simply call the function and it will automatically ask for the file cleaning is required on.<br>
```bash
data1, data2,data3,data4 = sd.saleem_data_cleaner()
```
data1 returns the dataframe without any mismatch.<br>
data2 returns the cleaned dataframe.<br>
data3 returns selected features from the dataframe.<br>
data4 returns the locals(functions).

