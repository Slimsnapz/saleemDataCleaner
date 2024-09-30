
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, Normalizer, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def saleem_data_cleaner():
    
    '''
    Functions available in the saleem_data_cleaner:
    - ##file_loader()##: Loads a CSV or Excel file based on user input.
    - check_data_shape(data): Displays the number of samples and features in the dataset.
    - cat_num_splitter(data): Splits the dataset into categorical and numerical features.
    - check_data_info(data): Displays the dataset information (data types, missing values, etc.).
    - feature_inspec(data): Inspects the first 8 values of each feature and checks for data type mismatches.
    - data_type_mismatch(data): Rectifies data type mismatches by converting object types with numeric values to appropriate data types and handles white spaces.
    - unique_labels(data): Identifies and displays the unique labels in all categorical features of the dataset.
    - missing_value(data): Checks for missing values in the dataset and returns the list of columns with missing values.
    - check_duplicates(data): Checks for duplicate rows in the dataset and returns a boolean indicating if duplicates are found.
    - duplicate_dropper(data): Drops duplicate rows from the dataset and updates the dataset.
    - missing_value_treater(fea_name, data): Treats missing values by filling columns with less than 5% missing data or drops columns with more than 5% missing values.
    - indicate_target(data): Identifies and separates the target variable (label) from the dataset. Returns the target column name, dataset without the target, and the encoded target variable.
    - feature_acc_label(label_dict, data): Prompts the user to provide labels for categorical features and categorizes them as nominal or ordinal.
    - label_mismatch(dic, data): Checks for mismatches between unique values in categorical features and the provided labels.
    - data_transformation(norminal, ordinal, cat_data): Transforms nominal features into one-hot encoded format and ordinal features into label-encoded format.
    - in_quantile_rng(data): Checks for outliers in numerical features and calculates the upper and lower limits using the interquartile range.
    - skew_check(data): Checks the skewness of numerical features.
    - check_data_balance(data): Checks the distribution of categorical features.
    - check_data_stat_behave(data): Provides a statistical summary (mean, std, min, max, etc.) of the numerical features.
    - is_need_to_normalize(data): Determines if normalization is required for numerical features.
    - normalizer(data): Normalizes the numerical features using the L2 normalization.
    - is_need_to_rescale(data): Determines if rescaling is required for numerical features based on range differences.
    - scaler(data): Scales numerical features using MinMaxScaler.
    - is_need_to_standardize(data): Determines if standardization is required based on the standard deviation of numerical features.
    - standardize(data): Standardizes numerical features using StandardScaler.
    
    - fea_sec_meth(data): Prompts the user to select a feature selection method: Univariate Feature Selection (UFS), Recursive Feature Elimination (RFE), or Principal Component Analysis (PCA).
    Returns the selected features and the accuracy of the model.
    
    - univariate_feature_selection(data, k=10): Performs Univariate Feature Selection using Logistic Regression and SelectKBest to select 'k' top features based on the highest scores.
    Returns a list of selected features and model accuracy.
    
    - recursive_feature_elimination(data, n_features_to_select=None): Performs Recursive Feature Elimination (RFE) to select a specified number of features.
    Returns a list of selected features and model accuracy.
    
    - pca_feature_selection(data, n_components=None): Performs Principal Component Analysis (PCA) to reduce feature dimensionality.
    Returns the explained variance ratio and model accuracy.
    '''

    # Function to load a CSV or Excel file  
    def file_loader():
        """
        file_loader: Loads a dataset from a CSV or Excel file.
        Prompts the user for the file name and file extension type.
        Returns the loaded dataset or an error message if the file doesn't exist.
    
        Returns:
        DataFrame: The loaded dataset if successful.
        None: If the file does not exist or if an unsupported file type is provided.
        """
        # Prompt user for dataset name and file extension type
        data = input('Input dataset name (without extension): ')
        file_type = input('Input file extension type (csv or xlsx): ').lower()  # Convert to lowercase for consistency
    
        # Build the full file name
        file_name = f"{data}.{file_type}"
    
        # Check if the file exists
        if not os.path.exists(file_name):
            print(f"Error: File '{file_name}' does not exist in the directory.")
            return None
    
        # Load the dataset based on the file extension
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_name)
            elif file_type == 'xlsx':
                df = pd.read_excel(file_name)
            else:
                print('Unsupported file extension. Please use either "csv" or "xlsx".')
                return None
            return df
        except Exception as e:
            print(f"An error occurred while loading the file: {str(e)}")
            return None



    def check_data_shape(data):
        """
        check_data_shape: Displays the number of samples (rows) and features (columns) in the dataset.
        Input: DataFrame (data)
        """
        shape = data.shape
        row = shape[0]
        features = shape[1]
        print(f'The data has {row} samples with {features} features')


    # Function to split the dataset into categorical and numerical features    
    def cat_num_splitter(data):
        """
        cat_num_splitter: Splits the dataset into categorical and numerical features.
        Input: DataFrame (data)
        Output: Two DataFrames - one for categorical features and one for numerical features.
        """
        cols = list(data.columns)
        cat_cols = []
        num_cols = []
        
        # Separate categorical and numerical columns based on data types
        for col in cols:
            if data[col].dtype == 'int64' or data[col].dtype == 'int32' or data[col].dtype == 'float64':
                num_cols.append(col)
            else:
                cat_cols.append(col)
        
        cat_df = data[cat_cols]
        num_df = data[num_cols]
        
        # returns categorical and numerical data
        print('*******************************************************************************************************************')
        print(f'The data has {len(cat_df.columns)} categorical features and {len(num_df.columns)} numerical features present in it.')
        print('*******************************************************************************************************************')
        return cat_df, num_df



    # Function to display general information about the dataset
    def check_data_info(data):
        print('*****************************************DATASET INFO**************************************************************')
        print(data.info())


    # Function to inspect each feature and check for data type mismatches
    def feature_inspec(data):
    
        """
        feature_inspec: Inspects the first 8 values of each feature and identifies data type mismatches.
        Input: DataFrame (data)
        Output: List of features with mismatched data types (e.g., numeric values stored as objects and vice versa).
        """   
        print('****************************ANALYZING FIRST 8 SAMPLE DATA OF EACH FEATURE*****************************************')
        mismatch = False
        mismatch_list = []
        for col in data.columns:
            type = data[col].dtype
            content = list(data[col])
            print(f'{col} contains {content[0:8]} and is currently of the {type} datatype')
            print('**************************************************************************************************************')
            for val in data[col]:
                if data[col].dtype == 'object' and val.isnumeric():
                    mismatch = True
                    mismatch_list.append(col)
                    break
        print(f'********************************DATATYPE MISMATCH IDENTIFIED AT {mismatch_list}')
        return mismatch_list
                    
    def data_type_mismatch(data):
        """
        data_type_mismatch: Identifies and rectifies data type mismatches by converting object columns with numeric values 
        to their appropriate types (e.g., float), while handling white spaces.
        Input: DataFrame (data)
        Output: DataFrame with corrected data types.
        """
        print('***********************************RECTIFYING DATA TYPE MISMATCH AND WHITE SPACES IN DATASET**************************************')
        for col in data.columns:
            for val in data[col]:
                if data[col].dtype == 'object' and val.isnumeric():
                    data[col] = data[col].replace(' ', pd.NA)    
                    # Convert the column to float
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    print(f'*************************DATATYPE MISMATCH AT COLUMN {col} HAS BEEN RECTIFIED SUCCESSFULLY************************')
                    break
        return data


    # Function to identify unique labels in categorical features
    def unique_labels(data):
        """
        unique_labels: Identifies and displays the unique labels in all categorical features of the dataset.
        Input: DataFrame (data)
        Output: Prints the unique labels in each categorical feature.
        """
        high_labels = []
        cols = list(cat_df.columns)
        print('***********************ANALYZING ALL CATEGORICAL FEATURES IN THE DATASETS AND IDENTIFYING ITS UNIQUE LABELS********************************')
        for col in cols:
            if len(data[col].unique()) > 20:
                should_drop = input(f'{col} unique labels exceeds 20. recommend it dropped. Yes/No: ').lower()
                if should_drop == 'yes':
                    high_labels.append(col)
                else:
                    continue
                if col not in high_labels: 
                    print(f"{col} has {len(data[col].unique())} unique features which are/is {data[col].unique()}")
        data = data.drop(high_labels, axis=1)
        return data

    # Function to check for missing values in the dataset
    def missing_value(data):
        """
        missing_value: Checks for missing values in the dataset.
        Input: DataFrame (data)
        Output: List of columns with missing values.
        """
        cols = list(data.columns)
        col_to_treat = []
        missing_values = False
        for col in cols:
            if data[col].isnull().sum() == 0:
                pass
            elif data[col].isnull().sum() > 0:
                    missing_values = True 
                    print('************************************************************************************')
                    print(f'{col} has {data[col].isnull().sum()} missing values.')
                    col_to_treat.append(col)
        if not missing_values:
            print('there are no missing values in the dataset.............................................')
        return col_to_treat


    # Function to check for duplicates in the dataset
    def check_duplicates(data):
        """
        check_duplicates: Checks for duplicate rows in the dataset.
        Input: DataFrame (data)
        Output: Boolean indicating whether duplicates were found.
        """
        if data.duplicated().sum() > 0:
            print('There are duplicated data instances in the dataset.')
            return True
        else:
            print('No duplicated data instances were found.')
            return False

    # Function to drop duplicate rows from the dataset
    def duplicate_dropper(data):
        """
        duplicate_dropper: Drops duplicate rows from the dataset.
        Input: DataFrame (data)
        Output: Updated DataFrame without duplicate rows.
        """
        print('Dropping duplicate samples from the dataset...')
        data.drop_duplicates(inplace=True)
        print('Duplicates have been successfully removed.')
    
   
    # Function to treat missing values in the dataset    
    def missing_value_treater(fea_name,data):
        """
        missing_value_treater: Treats missing values in the dataset.
        Columns with <= 5% missing values are backfilled, and columns with > 5% missing values are dropped.
        Input: List of feature names (fea_name), DataFrame (data)
        Output: DataFrame with missing values treated.
        """
        print("Treating missing values in the dataset...")
        for col in fea_name:
            percentage = (data[col].isnull().sum()/data[col].shape[0]) * 100
            if percentage <= 5:
                data[col] = data[col].bfill()
                print(f'Missing values in {col} were successfully backfilled.')
            else:
                data.drop(col, axis=1, inplace=True)
                print(f'{col} had more than 5% missing values and was dropped.')
        return data

    def indicate_target(data):
        """
        indicate_target: Identifies and separates the target variable from the dataset.
        Prompts the user to specify the target feature, and separates it from the dataset.
        Converts the target to numerical format if it's categorical.
        Returns the target column name, dataset without the target, and the target variable.
        """
        target_variable = pd.DataFrame()  # Initialize an empty DataFrame for the target variable
        x = True
        
        while x:
            target_fea = input('Indicate the Target feature: ')
            if target_fea in list(data.columns):
                x = False  # Exit the loop if target feature is found
                data_without_target = data.drop(target_fea, axis=1)  # Remove target column from the dataset
                target_var = data[target_fea]  # Extract the target column
                
                # If the target is categorical, encode it
                if target_var.dtype == 'object':
                    encoder = LabelEncoder()
                    ord_encoder = encoder.fit_transform(target_var)
                    target_var = ord_encoder
                    target_variable[target_fea] = target_var  # Store the encoded target variable in a DataFrame
                    
                # If the target is numeric, store it as is
                elif target_var.dtype == 'int' or target_var.dtype == 'float':
                    target_variable = target_var
                    
            else:
                print(f'{target_fea} not present in dataset. Check the case use and spelling')
        
        return target_fea, data_without_target, target_variable


    
    def feature_acc_label(label_dict, data):
        """
        feature_acc_label: Prompts the user to provide labels for categorical features.
        Organizes features into nominal and ordinal categories.
        Updates the label dictionary with the provided labels.
        Returns the nominal and ordinal categorical features.
        """
        fea_label_dict = {}
        norminal_cat_data = []
        ordinal_cat_data = []
        print('Please provide the appropriate data labels for all your categorical data.')
        
        for feature_name in data.columns:
            print(f"Feature: {feature_name}")
            
            # Prompt the user for labels
            if not feature_name == 'customerID':
                label_input = input(f'Indicate labels for "{feature_name}" (use commas to separate labels, or press Enter to skip): ')
                
                if label_input.strip():  # Check if the user provided labels
                    label = [lbl.strip() for lbl in label_input.split(',')]
                    fea_label_dict[feature_name] = label
                else:
                    print(f'No labels provided for "{feature_name}". Moving to the next feature.')
                    fea_label_dict[feature_name] = list(cat_df[feature_name].unique())
                    
                
                # Prompt the user to specify if the feature is nominal or ordinal
                categorical = input(f'Is "{feature_name}" nominal or ordinal? (default is nominal): ').lower()
            
                if categorical == 'ordinal':
                    ordinal_cat_data.append(feature_name)
                    
                else:
                    norminal_cat_data.append(feature_name)  # Default to nominal if not specified 
                       
                
                # Update the label dictionary
                label_dict.update(fea_label_dict)
    
        print("Processing completed..................................................................................................................")
        return norminal_cat_data, ordinal_cat_data



    def label_mismatch(dic,data):
        """
        label_mismatch: Checks for mismatches between unique values in categorical features and the provided labels.
        If any mismatches are found, they are displayed.
        """
        data_mismatch = False
        for col in list(data.columns):
            for key,val in zip(dic.keys(), dic.values()): 
                if key == col:
                    data_label = list(data[col].unique())
                    for i in data_label:
                        if i in val:
                            pass
                        elif i not in val:
                            print(f'unique label {i} found in feature {col}')
                            data_mismatch = True
        if data_mismatch == False:
            print('there are no mismatch found in the dataset for the categorical columns........................................................')

    
    def in_quantile_rng(data):
        """
        in_quantile_rng: Checks for outliers in numerical features using interquartile range (IQR).
        Calculates the upper and lower limits for each feature and identifies potential outliers.
        """
        print('CHECKING FOR OUTLIERS IN NUMERICAL FEATURES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        outliers = False
        for col in data.columns:
            q3, q1 = np.percentile(data[col], [75, 25]) # Calculate Q3 (75th percentile) and Q1 (25th percentile)
            iqr = q3 - q1 # Calculate the interquartile range (IQR)
            ul = q3 + 1.5 * iqr # Upper limit for outliers
            ll = q1 - 1.5 * iqr # Lower limit for outliers
            print(f'{col}: Upper limit = {ul}, Lower limit = {ll}')           
            if col != 'customerID':
                print('--------------------------------------------------------------------------------------------------------------------------')
                print(f'outlier statistics for {col} feature')
            
                print(f'upper limit is {ul}')
                print(f'lower limit is {ll}')
            
                total_outliers = len(num_df.loc[num_df[col] < ll, col]) + len(num_df.loc[num_df[col] > ul, col])
                percent_outliers = (total_outliers/len(num_df.index)) * 100
                print(f'percentage of outliers in {col} is {percent_outliers}')
                if percent_outliers > 5:
                    outliers = True
                    print(f'heavy presence of ouliers at {col} recommend treating it')
                print('---------------------------------------------------------------------------------------------------------------------------')
        print('<<<<<<<<< END OF OUTLIER ANALYSIS >>>>>>>>>>>>>>>>>>>>')
        print('...................................................................................................................................')

    
    # Function to check the skewness of numerical features
    def skew_check(data):
        """
        skew_check: Checks the skewness of each numerical feature in the dataset.
        Skewness measures the asymmetry of the distribution.
        """
        for col in data.columns:
            skewness = data[col].skew() # Calculate the skewness of the feature
            print('CHECKING SKEWNESS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(f'the feature {col} is {skewness} skewed......................................................................................')

    
    # Function to check the balance of categorical features
    def check_data_balance(data):
        """
        check_data_balance: Checks the distribution (balance) of categorical features in the dataset.
        """
        data_balance = []
        cols = list(data.columns)
        for col in cols:
            if col != "customerID":
                counts = data[col].value_counts() # Get value counts for the categorical feature
                data_balance.append(counts)
    
        print(data_balance)


    def data_transformation(norminal, ordinal, cat_data):
        """
        data_transformation: Transforms nominal features into one-hot encoded format and ordinal features into label-encoded format.
        Returns the transformed data with encoded categorical features.
        """
        transformed_data = pd.DataFrame()
        norm_df = cat_data[norminal]
        ord_df = cat_data[ordinal]
        norm_encoded = pd.get_dummies(norm_df).astype(int)
        transformed_data = norm_encoded
        encoder = LabelEncoder()
        for fea in ord_df.columns:
            ord_encoder = encoder.fit_transform(cat_data[fea])
            transformed_data[fea] = ord_encoder 

        return transformed_data

    
    # Function to display statistical behavior of numerical features
    def check_data_stat_behave(num_df):
        """
        check_data_stat_behave: Provides statistical summary (mean, std, min, max, etc.) of numerical features.
        """
        print('checking satistical behaviour of the numerical features in dataset')
        print(num_df.describe()) # Display statistical summary of numerical features

    
    # Function to determine if normalization is needed based on the feature range
    def is_need_to_normalize(data):
        """
        is_need_to_normalize: Determines if normalization is required based on the range of numerical features.
        """
        need_to_norm = []
        for col in data.columns:
            feature_range = (data.describe()[col]['max'] - data.describe()[col]['min']) / 2
            need_to_norm.append(feature_range)
        for i in need_to_norm:
            range_offset = data.describe()[col]['mean'] * 0.05  # 5% of mean as the range offset
            if i > range_offset:
                print('You need to normalize the data.')
                return True
        print('The features are normalized; normalization may not be necessary.')
        
    # Function to normalize numerical features using L2 normalization
    def normalizer(data):
        """
        normalizer: Normalizes numerical features using the L2 normalization method.
        """
        print('NORMALIZING NUMERICAL DATA IN PROGRESS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        numerical_to_nrmz = data.values          
        scaler = Normalizer().fit(numerical_to_nrmz) # Fit the Normalizer
        xnumerical_to_nrmz = scaler.transform(numerical_to_nrmz) # Normalize the data    
        numerical_to_nrmz_df = pd.DataFrame(data=numerical_to_nrmz, columns=list(data.columns))

        print('NORMALIZATION OF NUMERICAL DATA COMPLETED................................................................................................')
        print('.........................................................................................................................................')

        
        return numerical_to_nrmz_df
        
    # Function to determine if rescaling is needed based on range differences
    def is_need_to_rescale(data):
        """
        is_need_to_rescale: Determines if rescaling is needed based on the range differences between features.
        """
        ranges = []
        for col in data.columns:
            feature_range = data.describe()[col]['max'] - data.describe()[col]['min']
            ranges.append(feature_range)
        
        differences = [abs(ranges[i] - ranges[j]) for i in range(len(ranges)) for j in range(i + 1, len(ranges))]
        
        for diff in differences:
            if diff > 100:
                print("Consider rescaling the data.")
                return True
        
        print("Rescaling may not be necessary.")
        return False


    # Function to scale numerical features using MinMaxScaler
    def scaler(data):
        """
        scaler: Scales numerical features using MinMaxScaler.
        """
        print('SCALING NUMERICAL DATA IN PROGRESS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
    
        # Fit the scaler to the data and transform the data
        scaled_data = scaler.fit_transform(data) # Scale the data
        
        # Convert the normalized data back to a DataFrame with the same column names
        scaled_df = pd.DataFrame(data=scaled_data, columns=data.columns)
        
        print('SCALING OF NUMERICAL DATA COMPLETED................................................................................................')
        print('...................................................................................................................................')
        return scaled_df

  
    # Function to determine if standardization is needed based on standard deviation
    def is_need_to_standardize(data):
        """
        is_need_to_standardize: Determines if standardization is required based on the standard deviation of numerical features.
        """
        for col in data.columns:
            std = data.describe()[col]['std']  # Calculate the standard deviation of the feature
            if std > 1 or std < 0:
                print('Standardization may be required.')
                return True
        print('Standardization may not be necessary.')


    # Function to standardize numerical features using StandardScaler
    def standardize(data):
        print('STANDARDIZING THE NUMERICAL FEATUES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        scaler = StandardScaler()
        num_to_sd = scaler.fit_transform(data)  # Standardize the data
        standardized_df = pd.DataFrame(data=num_to_sd, columns=data.columns)
        print('STANDARDIZATION OF NUMERICAL FEATURES COMPLETE....................................................................................')
        print('..................................................................................................................................')
        return standardized_df


    def fea_sec_meth():
        """
        fea_sec_meth: Prompts the user to select a feature selection method (UFS, RFE, or PCA) 
        and applies the selected method to the dataset. It returns the selected features and the model's accuracy.

        Returns:
        - features_selected: List of selected features.
        - accuracy: Accuracy score of the model using the selected features.
        """
        x = 0
        while x == 0:
            method = input('Select the feature selection method you want to use: UFS or RFE or PCA')
            def univariate_feature_selection(data, k=10):
                """
                Perform univariate feature selection using Logistic Regression as an estimator.
            
                Parameters:
                - data: pandas DataFrame containing the dataset.
                - n: int, the number of top features to select (default is 10).
            
                Returns:
                - selected_features: List of selected feature names.
                - accuracy: Accuracy score of the model using the selected features.
                """
                
                # Split data into features and target
                inputs = data.iloc[:, :-1]  # All columns except the last one
                targets = data.iloc[:, -1]   # The last column (assumed to be the target)
            
                # Initialize SelectKBest with f_classif as the scoring function
                select_k_best = SelectKBest(score_func=f_classif, k=k)
            
                # Define the logistic regression model
                model = LogisticRegression()
            
                # Create a pipeline that first selects features and then fits the model
                pipeline = Pipeline([
                    ('feature_selection', select_k_best),
                    ('logistic_regression', model)
                ])
            
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3, random_state=365)
            
                # Fit the pipeline on the training data
                pipeline.fit(X_train, y_train)
            
                # Predict on the test data
                y_pred = pipeline.predict(X_test)
            
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
            
                # Get the selected features
                selected_features = inputs.columns[select_k_best.get_support()]
            
                print(f"Selected features: {selected_features}")
                print(f"Model accuracy with selected features: {accuracy:.4f}")
            
                return list(selected_features), accuracy
        
            def recursive_feature_elimination(data, n_features_to_select=None):
                """
                Perform Recursive Feature Elimination (RFE) for feature selection using Logistic Regression as the estimator.
            
                Parameters:
                - data: pandas DataFrame containing the dataset.
                - target_column: str, the name of the target variable column.
                - n_features_to_select: int or None, the number of features to select (default is None, which means half of the features).
            
                Returns:
                - selected_features: List of selected feature names.
                - accuracy: Accuracy score of the model using the selected features.
                """
            
                # Split data into features and target
                inputs = data.iloc[:, :-1]  # All columns except the last one
                targets = data.iloc[:, -1]   # The last column (assumed to be the target)
            
                # Initialize the logistic regression model
                model = LogisticRegression()
            
                # Initialize RFE with the specified number of features to select
                rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
            
                # Fit RFE on the data
                rfe.fit(inputs, targets)
            
                # Get the selected features
                selected_features = inputs.columns[rfe.support_].tolist()
            
                # Split the data into training and testing sets based on selected features
                X_train, X_test, y_train, y_test = train_test_split(inputs[selected_features], targets, test_size=0.3, random_state=365)
            
                # Fit the model on the selected features
                model.fit(X_train, y_train)
            
                # Predict on the test data
                y_pred = model.predict(X_test)
            
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
            
                print(f"Selected features: {selected_features}")
                print(f"Model accuracy with selected features: {accuracy:.4f}")
            
                return selected_features, accuracy
        
        
            def pca_feature_selection(data, n_components=None):
                """
                Perform PCA for feature selection and use Logistic Regression as an estimator.
            
                Parameters:
                - data: pandas DataFrame containing the dataset.
                - target_column: str, the name of the target variable column.
                - n_components: int or None, the number of principal components to keep (default is None, which means all components).
            
                Returns:
                - explained_variance_ratio: The variance explained by each of the selected components.
                - accuracy: Accuracy score of the model using the selected components.
                """
            
                # Split data into features and target
                inputs = data.iloc[:, :-1]  # All columns except the last one
                targets = data.iloc[:, -1]   # The last column (assumed to be the target)
            
                # Initialize PCA
                pca = PCA(n_components=n_components)
            
                # Define the logistic regression model
                model = LogisticRegression()
            
                # Create a pipeline that first applies PCA and then fits the model
                pipeline = Pipeline([
                    ('pca', pca),
                    ('logistic_regression', model)
                ])
            
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3, random_state=365)
            
                # Fit the pipeline on the training data
                pipeline.fit(X_train, y_train)
            
                # Predict on the test data
                y_pred = pipeline.predict(X_test)
            
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
            
                # Get the explained variance ratio of the selected components
                explained_variance_ratio = pca.explained_variance_ratio_
            
                print(f"Explained variance by each component: {explained_variance_ratio}")
                print(f"Model accuracy with selected components: {accuracy:.4f}")
            
                return explained_variance_ratio, accuracy
    
                
            if method.lower() == 'ufs':
                features_selected, accuracy = univariate_feature_selection(cleaned_df) 
                x = 1
            elif method.lower() == 'rfe':
                features_selected, accuracy = recursive_feature_elimination(cleaned_df)
                x = 1
            elif method.lower() == 'pca':
                features_selected, accuracy = pca_feature_selection(cleaned_df)
                x = 1
            else:
                print(f'{method} is not available, please input the available methods listed...........................')
        return features_selected, accuracy

    file = True

    while file:
        df = file_loader()
        
        if df is not None:  # If a valid dataframe is loaded
            file = False  # Exit the loop
    check_data_shape(df)
    check_data_info(df)
    mismatch = feature_inspec(df)
    if mismatch:
        df_no_mis_match = data_type_mismatch(df)
    else:
        df_no_mis_match = df
    col_to_treat = missing_value(df_no_mis_match)
    if col_to_treat:
        missing_value_treater(col_to_treat,df_no_mis_match)
    if check_duplicates(df_no_mis_match):
        duplicate_dropper(df_no_mis_match)
    target_name, data_without_target, target_var = indicate_target(df_no_mis_match)
    cat_df, num_df = cat_num_splitter(data_without_target)
    cat_df = unique_labels(cat_df)
    check_data_balance(cat_df)
    label_dict = {}
    norminal_cat_data = []
    ordinal_cat_data = []
    norminal_fea, ordinal_fea = feature_acc_label(label_dict, cat_df)
    label_mismatch(label_dict,cat_df)
    in_quantile_rng(num_df)
    skew_check(num_df)
    transformed_data = data_transformation(norminal_fea, ordinal_fea, cat_df)
    check_data_stat_behave(num_df)
    # Check if normalization is needed
    if is_need_to_normalize(num_df):
        scaled_num_df = normalizer(num_df)
    else:
        scaled_num_df = num_df  # No normalization needed, use original data  
    # Check if rescaling is needed
    if is_need_to_rescale(scaled_num_df):
        scaled_num_df = scaler(scaled_num_df)
    else:
        scaled_num_df = scaled_num_df  # No rescaling needed, use previous data 
    # Check if standardization is needed
    if is_need_to_standardize(scaled_num_df):
        scaled_num_df = standardize(scaled_num_df)
    else:
        pass


    cleaned_df = pd.concat([transformed_data, scaled_num_df, target_var], axis=1)
    features_selected, accuracy = fea_sec_meth()
    cleaned_wt_sel_fea = pd.DataFrame(columns=features_selected, data=cleaned_df[features_selected])
    
    
    return df_no_mis_match, cleaned_df, cleaned_wt_sel_fea, locals()
