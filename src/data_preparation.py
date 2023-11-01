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
    def __init__(self, train, test):
        self.train = train.copy()
        self.test = test.copy()

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

    def impute_train_test_numerical(self):

        # Imputation of Electric range
        self.train['Electric_range_(km)'].fillna(0, inplace=True)
        self.test['Electric_range_(km)'].fillna(0, inplace=True)

        # Imputation of Ec
        self.train["ec_(cm3)"].fillna(0, inplace=True)
        self.test['ec_(cm3)'].fillna(0, inplace=True)

        #Imputation of Fuel Consumption

        self.train.loc[(self.train["Fuel_consumption_"].isna()) & (self.train["Ft"] == 'ELECTRIC'), "Fuel_consumption_"] = 0
        self.test.loc[(self.test["Fuel_consumption_"].isna()) & (self.test["Ft"] == 'ELECTRIC'), "Fuel_consumption_"] = 0

        mean_FC_by_Cn = self.train.groupby('Cn')['Fuel_consumption_'].transform('mean')
        self.train['Fuel_consumption_'].fillna(mean_FC_by_Cn, inplace=True)
        self.train["Fuel_consumption_"].fillna(self.train["Fuel_consumption_"].mean(), inplace=True)

        mean_FC_by_Cn = self.test.groupby('Cn')['Fuel_consumption_'].transform('mean')
        self.test['Fuel_consumption_'].fillna(mean_FC_by_Cn, inplace=True)
        self.test["Fuel_consumption_"].fillna(self.test["Fuel_consumption_"].mean(), inplace=True)

        #Imputation of Mt
        self.train.loc[self.train["Mt"].isna(), "Mt"] = self.train.loc[self.train["Mt"].isna()]["m_(kg)"]
        self.train["Mt"].fillna(self.train["Mt"].mean(), inplace=True)

        self.test.loc[self.test["Mt"].isna(), "Mt"] = self.test.loc[self.test["Mt"].isna()]["m_(kg)"]
        self.test["Mt"].fillna(self.test["Mt"].mean(), inplace=True)

        #Imputation of At1 (mm)
        mean_At1_by_Cn = self.train.groupby('Cn')['At1_(mm)'].transform('mean')
        self.train['At1_(mm)'].fillna(mean_At1_by_Cn, inplace=True)
        self.train["At1_(mm)"].fillna(self.train["At1_(mm)"].mean(), inplace=True)

        mean_At1_by_Cn = self.test.groupby('Cn')['At1_(mm)'].transform('mean')
        self.test['At1_(mm)'].fillna(mean_At1_by_Cn, inplace=True)
        self.test["At1_(mm)"].fillna(self.test["At1_(mm)"].mean(), inplace=True)

        # Imputation of At2 (mm)
        mean_At2_by_Cn = self.train.groupby('Cn')['At2_(mm)'].transform('mean')
        self.train['At2_(mm)'].fillna(mean_At2_by_Cn, inplace=True)
        self.train["At2_(mm)"].fillna(self.train["At2_(mm)"].mean(), inplace=True)

        mean_At2_by_Cn = self.test.groupby('Cn')['At2_(mm)'].transform('mean')
        self.test['At2_(mm)'].fillna(mean_At2_by_Cn, inplace=True)
        self.test["At2_(mm)"].fillna(self.test["At2_(mm)"].mean(), inplace=True)

        #Imputation of m (kg)
        mean_m_by_Cn = self.train.groupby('Cn')['m_(kg)'].transform('mean')
        self.train['m_(kg)'].fillna(mean_m_by_Cn, inplace=True)
        self.train["m_(kg)"].fillna(self.train["m_(kg)"].mean(), inplace=True)

        mean_m_by_Cn = self.test.groupby('Cn')['m_(kg)'].transform('mean')
        self.test['m_(kg)'].fillna(mean_m_by_Cn, inplace=True)
        self.test["m_(kg)"].fillna(self.test["m_(kg)"].mean(), inplace=True)

        #Imputation of W
        mean_W_by_Cn = self.train.groupby('Cn')['W_(mm)'].transform('mean')
        self.train['W_(mm)'].fillna(mean_W_by_Cn, inplace=True)
        self.train["W_(mm)"].fillna(self.train["W_(mm)"].mean(), inplace=True)

        mean_W_by_Cn = self.test.groupby('Cn')['W_(mm)'].transform('mean')
        self.test['W_(mm)'].fillna(mean_W_by_Cn, inplace=True)
        self.test["W_(mm)"].fillna(self.test["W_(mm)"].mean(), inplace=True)

        #Imputation of ep
        mean_ep_by_Cn = self.train.groupby('Cn')['ep_(KW)'].transform('mean')
        self.train['ep_(KW)'].fillna(mean_ep_by_Cn, inplace=True)
        self.train["ep_(KW)"].fillna(self.train["ep_(KW)"].mean(), inplace=True)

        mean_ep_by_Cn = self.test.groupby('Cn')['ep_(KW)'].transform('mean')
        self.test['ep_(KW)'].fillna(mean_ep_by_Cn, inplace=True)
        self.test["ep_(KW)"].fillna(self.test["ep_(KW)"].mean(), inplace=True)

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

        print("Valeurs manquantes catégorielles imputées ✅")

    def prepare_data(self):
        self.remove_train_nan()
        self.remove_test_nan()
        self.rename_columns()
        self.get_type_list()
        self.impute_train_test_numerical()
        self.impute_train_test_categorical()
        return self.train, self.test

