import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


class DataPreparation:
    def __init__(self, train, test):
        self.train = train.copy()
        self.test = test.copy()

    def get_variable_correlation(self, variable):
        correlation_vector = self.train[self.col_numericals].corr()[variable][:]
        correlation_vector = np.abs(correlation_vector)
        correlation_vector = correlation_vector.sort_values(ascending=False)[1:]

        return correlation_vector

    def get_nan_table(self):
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
        percentage = (self.train.isna().sum().sum() / self.train.shape[0]) * 100
        self.columns_to_delete = [
            col
            for col in self.train.columns
            if col not in "Electric range (km)" and percentage[col] >= 40
        ]
        self.train.drop(columns=self.columns_to_delete, inplace=True)

    def remove_test_nan(self):
        self.test.drop(columns=self.columns_to_delete, inplace=True)

    def get_type_list(self):
        self.col_categoricals = self.train.select_dtypes(
            include=["object"]
        ).columns.to_list()

        self.col_numericals = self.train.select_dtypes(
            exclude=["object"]
        ).columns.to_list()
        self.train["date"] = pd.to_datetime(self.train["Date of registration"])
        self.col_numericals.remove("ID")

    def remove_train_outliers(self):
        self.train = self.train.loc[self.train["W (mm)"] >= 1500]
        return self.train

    def remove_test_outliers(self):
        self.test = self.test.loc[self.test["W (mm)"] >= 1500]
        return self.test

    def impute_train_test_numerical(self):
        var_to_impute = self.col_numericals.copy()
        var_to_impute.remove("Electric range (km)")
        var_to_impute.remove("Ewltp (g/km)")

        X_train, X_test = self.train[var_to_impute], self.test[var_to_impute]

        imputer = IterativeImputer(max_iter=5, random_state=0)
        imputer.fit(X_train)

        X_train_imputed = imputer.transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        self.train.loc[:, var_to_impute] = X_train_imputed
        self.test.loc[:, var_to_impute] = X_test_imputed

        self.train.loc[self.train["Fuel consumption "] <= 0, "Fuel consumption "] = 0
        self.test.loc[self.test["Fuel consumption "] <= 0, "Fuel consumption "] = 0

    def impute_categorical(self):
        nan_table = self.get_nan_table()
        var_to_impute = nan_table[
            (self.train["type"] == "Catégorielle")
            and (self.train["Pourcentage de valeurs manquantes"] > 0)
        ].columns.to_list()

        for var in var_to_impute:
            self.train[var] = self.train.groupby("Pclass", group_keys=False)[var].apply(
                lambda x: x.fillna(x.mode()[0])
            )
