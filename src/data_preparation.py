import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

class customdataset:
    """
    This module allows to convert the data into torch-readable format.
    """
    def __init__(self, data):
        self.data = data

    def nbr_var(self):
        """
        Retrieves the number of features in the dataset.

        Return:
            - nbr_var (int) : Feature's number.
        """
        nbr_var = self.data.shape[1]
        return (nbr_var)


    def __len__(self):
        """
        Retrieves the number of the observation in the dataset.

        Return:
            - length (int) : number of the observation
        """
        length = len(self.data)
        return (length)

    def __getitem__(self, idx):

        if self.data.ndim == 2:  # Vérifiez si les données sont bien 2D
            current_sample = self.data[idx, :]
            obs = torch.tensor(current_sample, dtype=torch.float)

        elif self.data.ndim == 1:  # Si les données sont 1D, traitez-les comme telles
            current_sample = self.data[idx]
            obs = torch.tensor([current_sample], dtype=torch.float)

        return obs

class DataPreparation:
    def __init__(self, train, test, neural_networks = False, target = None):
        self.train = train.copy()
        self.test = test.copy()
        self.target = target
        self.neural_networks = neural_networks

    def get_variable_correlation(self, variable):
        correlation_vector = self.train[self.col_numericals].corr()[variable][:]
        correlation_vector = np.abs(correlation_vector)
        correlation_vector = correlation_vector.sort_values(ascending=False)[1:]

        return (correlation_vector)

    def get_nan_table(self):
        nan_table = pd.DataFrame(columns=["Variable", "Pourcentages de valeurs manquantes", "Type"])

        for col in self.train:
            pcentage = (self.train[col].isna().sum() / self.train.shape[0]) * 100
            type = 'Numérique' if col in self.col_numericals else 'Catégorielle'
            nan_table.loc[len(nan_table)] = [col, pcentage, type]

        nan_table = nan_table.sort_values(by=["Pourcentages de valeurs manquantes"], ascending=False).reset_index(
            drop=True)

        return nan_table

    def remove_train_nan(self):
        percentage = (self.train.isna().sum() / self.train.shape[0]) * 100
        self.columns_to_delete = [col for col in self.train.columns if
                                  col != "Electric range (km)" and percentage[col] >= 40]
        self.columns_to_delete.extend(["r", "Status", "IT", 'Date of registration', "Mp"])
        self.train.drop(columns=self.columns_to_delete, inplace=True)
        print("Valeurs manquantes du train supprimées ✅")

    def remove_test_nan(self):
        self.test.drop(columns=self.columns_to_delete, inplace=True)
        print("Valeurs manquantes du test supprimées ✅")

    def rename_columns(self):
        self.train.columns = self.train.columns.str.replace(' ', '_')
        self.test.columns = self.test.columns.str.replace(' ', '_')
        print("Variables renommées ✅")

    def get_type_list(self):
        self.col_categoricals = self.train.select_dtypes(include=['object']).columns.to_list()
        self.col_numericals = self.train.select_dtypes(exclude=["object"]).columns.to_list()
        self.col_numericals.remove("ID")

    def impute_by_mean(self, var_to_impute, var_mean):
        mean_train = self.train.groupby(var_mean)[var_to_impute].transform('mean')
        self.train[var_to_impute].fillna(mean_train, inplace=True)
        self.train[var_to_impute].fillna(self.train[var_to_impute].mean(), inplace=True)

        mean_test = self.test.groupby(var_mean)[var_to_impute].transform('mean')
        self.test[var_to_impute].fillna(mean_test, inplace=True)
        self.test[var_to_impute].fillna(self.test[var_to_impute].mean(), inplace=True)

    def impute_train_test_numerical(self):

        # Imputation of Electric range
        self.train['Electric_range_(km)'].fillna(0, inplace=True)
        self.test['Electric_range_(km)'].fillna(0, inplace=True)

        # Imputation of Ec
        self.train["ec_(cm3)"].fillna(0, inplace=True)
        self.test['ec_(cm3)'].fillna(0, inplace=True)

        #Imputation of Mt
        self.train.loc[self.train["Mt"].isna(), "Mt"] = self.train.loc[self.train["Mt"].isna()]["m_(kg)"]
        self.train["Mt"].fillna(self.train["Mt"].mean(), inplace=True)

        self.test.loc[self.test["Mt"].isna(), "Mt"] = self.test.loc[self.test["Mt"].isna()]["m_(kg)"]
        self.test["Mt"].fillna(self.test["Mt"].mean(), inplace=True)

        #Imputation of Fuel Consumption

        self.train.loc[(self.train["Fuel_consumption_"].isna()) & (self.train["Ft"] == 'ELECTRIC'), "Fuel_consumption_"] = 0
        self.test.loc[(self.test["Fuel_consumption_"].isna()) & (self.test["Ft"] == 'ELECTRIC'), "Fuel_consumption_"] = 0

        self.impute_by_mean("Fuel_consumption_", "Cn")

        #Imputation of At1 (mm)
        self.impute_by_mean("At1_(mm)", "Cn")

        # Imputation of At2 (mm)
        self.impute_by_mean("At2_(mm)", "Cn")

        #Imputation of m (kg)
        self.impute_by_mean("m_(kg)", "Cn")

        #Imputation of W
        self.impute_by_mean("W_(mm)", "Cn")

        #Imputation of ep
        self.impute_by_mean("ep_(KW)", "Cn")

        print("Valeurs manquantes numériques imputées ✅")

    def impute_train_test_categorical(self):
        #Imputation of 'Cn'
        mode_VFN_train = self.train.groupby('T')['Cn'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        self.train['Cn'] = self.train['Cn'].fillna(self.train['T'].map(mode_VFN_train))

        mode_VFN_test = self.test.groupby('T')['Cn'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        self.test['Cn'] = self.test['Cn'].fillna(self.test['T'].map(mode_VFN_test))

        # Impute 'Cn' par le mode de Cn si T est manquant
        self.train['Cn'].fillna(self.train['Cn'].mode()[0], inplace=True)

        self.test['Cn'].fillna(self.test['Cn'].mode()[0], inplace=True)


        # Imputation of 'VFN'
        mode_VFN_train = self.train.groupby('Cn')['VFN'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        self.train['VFN'] = self.train['VFN'].fillna(self.train['Cn'].map(mode_VFN_train))

        mode_VFN_test = self.test.groupby('Cn')['VFN'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        self.test['VFN'] = self.test['VFN'].fillna(self.test['Cn'].map(mode_VFN_test))

        # Impute 'VFN' with 'Va' if 'Cn' is missing
        self.train['VFN'] = self.train.apply(
            lambda row: row['Va'] if pd.isna(row['VFN']) and not pd.isna(row['Cn']) else row['VFN'],
            axis=1)

        self.test['VFN'] = self.test.apply(
            lambda row: row['Va'] if pd.isna(row['VFN']) and not pd.isna(row['Cn']) else row['VFN'],
            axis=1)

        # Impute 'VFN' with mode of 'VFN' if both 'Cn' and 'Va' are missing
        self.train['VFN'].fillna(self.train['VFN'].mode()[0], inplace=True)

        self.test['VFN'].fillna(self.test['VFN'].mode()[0], inplace=True)

        # Imputation des variables ayant moins de 1% de NaN
        for col in ['Tan', 'T', 'Va', 'Ve', 'Mk', 'Ct', 'Fm']:
            self.train[col].fillna(self.train[col].mode()[0],inplace=True)
            self.test[col].fillna(self.train[col].mode()[0], inplace=True)

        self.train.Ft = self.train.Ft.apply(lambda x: "PETROL" if x == "UNKNOWN" else x)
        self.train["Ewltp_(g/km)"] = self.train["Ewltp_(g/km)"].apply(lambda x: 0 if x < 0 else x)
        self.train.Ft = self.train.Ft.apply(lambda x: "NG" if x == "NG-BIOMETHANE" else x)
        self.train.Ft = self.train.Ft.apply(
            lambda x: "ELECTRIC/HYDROGEN" if x == "HYDROGEN" or x == "ELECTRIC" else x)
        self.train.Ft = self.train.Ft.apply(
            lambda x: "HYBRID" if x == "PETROL/ELECTRIC" or x == "DIESEL/ELECTRIC" else x)
        self.train.Ft = self.train.Ft.apply(lambda x: "BIOCARB" if x == "NG" or x == "E85" or x == "LPG" else x)

        self.test.Ft = self.test.Ft.apply(lambda x: "PETROL" if x == "UNKNOWN" else x)
        self.test.Ft = self.test.Ft.apply(lambda x: "NG" if x == "NG-BIOMETHANE" else x)
        self.test.Ft = self.test.Ft.apply(lambda x: "ELECTRIC/HYDROGEN" if x == "HYDROGEN" or x == "ELECTRIC" else x)
        self.test.Ft = self.test.Ft.apply(
            lambda x: "HYBRID" if x == "PETROL/ELECTRIC" or x == "DIESEL/ELECTRIC" else x)
        self.test.Ft = self.test.Ft.apply(lambda x: "BIOCARB" if x == "NG" or x == "E85" or x == "LPG" else x)

        print("Valeurs manquantes catégorielles imputées ✅")

    def tensor_transformation(self, X_train, y_train):
        train_customdataset = customdataset(np.array(X_train))
        self.train_dataloader = DataLoader(train_customdataset, batch_size=256)

        target_customdataset = customdataset(np.array(y_train))
        self.target_dataloader = DataLoader(target_customdataset, batch_size=256)

    def prepare_data(self):
        self.remove_train_nan()
        self.remove_test_nan()
        self.rename_columns()
        self.get_type_list()
        self.impute_train_test_numerical()
        self.impute_train_test_categorical()

        if self.neural_networks :
            X = self.train[self.col_numericals].drop(columns = [self.target])
            y = self.train[self.target]

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

            self.tensor_transformation(X_train, y_train)
            return X_train, X_val, y_train, y_val

        return self.train, self.test



