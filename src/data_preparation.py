import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


class DataPreparation:
    """
    DataPreparation class to prepare the data
    """

    def __init__(self, train, test):
        self.train = train.copy()
        self.test = test.copy()

    def get_variable_correlation(self, variable):
        """
        Get the correlation vector for a given variable.

        Parameters:
            variable (str): The name of the variable to calculate
            the correlation vector for.

        Returns:
            pd.Series: The correlation vector for the given variable,
            sorted in descending order.
        """
        correlation_vector = self.train[self.col_numericals].corr()[variable][:]
        correlation_vector = np.abs(correlation_vector)
        correlation_vector = correlation_vector.sort_values(ascending=False)[1:]

        return correlation_vector

    def get_nan_table(self):
        """
        Generates a table showing the percentage of missing values for each
        variable in the dataset.

        Returns:
            pandas.DataFrame: A DataFrame containing three columns: "Variable",
            "Pourcentages de valeurs manquantes", and "Type".
                - The "Variable" column contains the names of the variables
                in the dataset.
                - The "Pourcentages de valeurs manquantes" column contains
                the percentage of missing values for each variable.
                - The "Type" column indicates whether the variable
                is numerical or categorical.
        """
        nan_table = pd.DataFrame(
            columns=["Variable", "Pourcentages de valeurs manquantes", "Type"]
        )

        for col in self.train:
            pcentage = (self.train[col].isna().sum() / self.train.shape[0]) * 100
            type = "Numérique" if col in self.col_numericals else "Catégorielle"
            nan_table.loc[len(nan_table)] = [col, pcentage, type]

        nan_table = nan_table.sort_values(
            by=["Pourcentages de valeurs manquantes"], ascending=False
        ).reset_index(drop=True)

        return nan_table

    def remove_train_nan(self):
        """
        Removes columns with a high percentage of missing values from
        the train dataset.

        Returns:
            None
        """
        percentage = (self.train.isna().sum() / self.train.shape[0]) * 100

        self.columns_to_delete = [
            col
            for col in self.train.columns
            if col != "Electric range (km)" and percentage[col] >= 40
        ]
        self.columns_to_delete.extend(["r", "Status", "IT", "Mp"])
        self.train.drop(columns=self.columns_to_delete, inplace=True)
        print("Valeurs manquantes du train supprimées ✅")

    def remove_test_nan(self):
        """
        Remove nan values from the test dataframe.
        """
        self.test.drop(columns=self.columns_to_delete, inplace=True)
        print("Valeurs manquantes du test supprimées ✅")

    def remove_train_test_outliers(self):
        """
        Remove outliers from the train and test datasets based on the "W (mm)"
        column.
        """
        self.train = self.train.loc[self.train["W (mm)"] >= 1500]
        self.test = self.test.loc[self.test["W (mm)"] >= 1500]
        print("Outliers traités ✅")

    def get_type_list(self):
        """
        Get the list of column types in the dataset.

        This function retrieves the list of column types in the dataset
        and stores them in
        class variables `col_categoricals` and `col_numericals`.

        Parameters:
        - self: The instance of the class.

        Returns:
        None
        """
        self.col_categoricals = self.train.select_dtypes(
            include=["object"]
        ).columns.to_list()
        self.col_numericals = self.train.select_dtypes(
            exclude=["object"]
        ).columns.to_list()
        self.train["date"] = pd.to_datetime(self.train["Date of registration"])
        self.col_numericals.remove("ID")

    def impute_train_test_numerical(self):
        """
        Imputes missing values in the train and test datasets for
        numerical features.

        Parameters:
        - None

        Returns:
        - None
        """
        # Imputation of all numerical features except Eletric Range and Fuel Consumption
        var_to_impute = self.col_numericals.copy()
        var_to_impute.remove("Electric range (km)")
        var_to_impute.remove("Ewltp (g/km)")
        var_to_impute.remove("Fuel consumption ")

        X_train, X_test = self.train[var_to_impute], self.test[var_to_impute]

        imputer = IterativeImputer(max_iter=5, random_state=0)
        imputer.fit(X_train)

        X_train_imputed = imputer.transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        self.train.loc[:, var_to_impute] = X_train_imputed
        self.test.loc[:, var_to_impute] = X_test_imputed

        # Imputation of Electric range
        self.train["Electric range (km)"].fillna(0, inplace=True)
        self.test["Electric range (km)"].fillna(0, inplace=True)

        # Imputation of Fuel Consumption
        var_explicatives = self.get_variable_correlation(
            "Fuel consumption "
        ).index.to_list()
        var_explicatives.remove("Ewltp (g/km)")
        var_explicatives = var_explicatives[:3]

        var_full = var_explicatives.copy()
        var_full.append("Fuel consumption ")

        df_train = self.train[var_full].dropna(how="any")
        index_NAN_train = self.train[self.train["Fuel consumption "].isna()].index
        index_NAN_test = self.test[self.test["Fuel consumption "].isna()].index

        reg = LinearRegression().fit(
            df_train[var_explicatives], df_train["Fuel consumption "]
        )

        pred_train = reg.predict(self.train.loc[index_NAN_train, var_explicatives])
        pred_test = reg.predict(self.test.loc[index_NAN_test, var_explicatives])

        self.train.loc[index_NAN_train, "Fuel consumption "] = pred_train
        self.test.loc[index_NAN_test, "Fuel consumption "] = pred_test

        self.train.loc[self.train["Fuel consumption "] <= 0, "Fuel consumption "] = 0
        self.test.loc[self.test["Fuel consumption "] <= 0, "Fuel consumption "] = 0

        print("Valeurs manquantes numériques imputées ✅")

    def impute_train_test_categorical(self):
        """
        Imputes missing categorical values in the train and test datasets.

        Parameters:
            None

        Returns:
            None
        """
        # imputation 'Cn'-------------------------------------------------------------------------------------------------
        mode_VFN_train = self.train.groupby("T")["Cn"].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        self.train["Cn"] = self.train["Cn"].fillna(self.train["T"].map(mode_VFN_train))

        mode_VFN_test = self.test.groupby("T")["Cn"].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        self.test["Cn"] = self.test["Cn"].fillna(self.test["T"].map(mode_VFN_test))

        # Impute 'Cn' par le mode de Cn si T est manquant
        self.train["Cn"].fillna(self.train["Cn"].mode()[0], inplace=True)

        self.test["Cn"].fillna(self.test["Cn"].mode()[0], inplace=True)

        # Imputation 'VFN' ---------------------------------------------------------------------------------------------
        mode_VFN_train = self.train.groupby("Cn")["VFN"].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        self.train["VFN"] = self.train["VFN"].fillna(
            self.train["Cn"].map(mode_VFN_train)
        )

        mode_VFN_test = self.test.groupby("Cn")["VFN"].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )
        self.test["VFN"] = self.test["VFN"].fillna(self.test["Cn"].map(mode_VFN_test))

        # Impute 'VFN' with 'Va' if 'Cn' is missing
        self.train["VFN"] = self.train.apply(
            lambda row: row["Va"]
            if pd.isna(row["VFN"]) and not pd.isna(row["Cn"])
            else row["VFN"],
            axis=1,
        )

        self.test["VFN"] = self.test.apply(
            lambda row: row["Va"]
            if pd.isna(row["VFN"]) and not pd.isna(row["Cn"])
            else row["VFN"],
            axis=1,
        )

        # Impute 'VFN' with mode of 'VFN' if both 'Cn' and 'Va' are missing
        self.train["VFN"].fillna(self.train["VFN"].mode()[0], inplace=True)

        self.test["VFN"].fillna(self.test["VFN"].mode()[0], inplace=True)

        # drop de 'date of registration'
        self.train.drop(columns="Date of registration", inplace=True)
        self.test.drop(columns="Date of registration", inplace=True)

        # Imputation des variables ayant moins de 1% de NaN
        for col in ["Tan", "T", "Va", "Ve", "Mk", "Ct", "Fm"]:
            self.train[col].fillna(self.train[col].mode()[0], inplace=True)
            self.test[col].fillna(self.train[col].mode()[0], inplace=True)

        print("Valeurs manquantes catégorielles imputées ✅")

    def prepare_data(self):
        """
        Prepare the data for training and testing.

        This function performs the following operations:
        - Removes NaN values from the training dataset.
        - Removes NaN values from the testing dataset.
        - Removes outliers from the training and testing datasets.
        - Retrieves the list of data types in the dataset.
        - Imputes missing numerical values in the train and test datasets.
        - Imputes missing categorical values in the train and test datasets.

        Returns:
        - train (DataFrame): The preprocessed training dataset.
        - test (DataFrame): The preprocessed testing dataset.
        """
        self.remove_train_nan()
        self.remove_test_nan()
        self.remove_train_test_outliers()
        self.get_type_list()
        self.impute_train_test_numerical()
        self.impute_train_test_categorical()
        return self.train, self.test
