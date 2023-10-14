import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class DataPreparation:
    def __init__(self, df):
        self.df = df.copy()


    def get_type_list(self):
        self.col_categoricals = self.df.select_dtypes(include = ['object']).columns.to_list()
        self.col_numericals = self.df.select_dtypes(exclude = ["object"]).columns.to_list()
        self.df["date"] = pd.to_datetime(self.df["Date of registration"])


    def remove_outliers(self):
        pass



    def get_nan_table(self):
        nan_table = pd.DataFrame(columns=["Variable", "Pourcentages de valeurs manquantes", "Type"])

        for col in self.df:
            pcentage = (self.df[col].isna().sum() / self.df.shape[0]) * 100
            type = 'Numérique' if col in self.col_numericals else 'Catégorielle'
            nan_table.loc[len(nan_table)] = [col, pcentage, type]

        nan_table = nan_table.sort_values(by=["Pourcentages de valeurs manquantes"], ascending=False).reset_index(drop=True)

        return nan_table

    def remove_nan(self):
        percentage = (self.df.isna().sum().sum()/self.df.shape[0])*100
        columns_to_delete = [col for col in self.df.columns if col not in "Electric range (km)" and percentage[col] >= 40]
        self.df.drop(columns = columns_to_delete, inplace = True)

    def get_variable_correlation(self, variable):
        correlation_vector = self.df[self.col_numericals].corr()[variable][:]
        correlation_vector = np.abs(correlation_vector)
        correlation_vector = correlation_vector.sort_values(ascending=False)[1:]

        return(correlation_vector)

    def impute_numerical(self):

        nan_table = self.get_nan_table()
        var_to_impute = nan_table[(nan_table['Type'] == 'Numérique') & (nan_table["Pourcentages de valeurs manquantes"] > 0)]['Variable'].to_list()
        print(var_to_impute)
        for var in var_to_impute:
            index_NAN = self.df[self.df[var].isna()].index
            var_explicatives = []

            correlation_vector = self.get_variable_correlation(var)
            var_correlated = correlation_vector.index.to_list()

            for colonne in var_correlated:
                if self.df[colonne].loc[index_NAN].isna().sum() == 0 :
                    var_explicatives.append(colonne)
                    if len(var_explicatives) == 3 :
                        break

            print(var_explicatives)
            df_train = self.df.dropna(how='any')
            
            reg = LinearRegression().fit(df_train[var_explicatives], df_train[var])
            pred = reg.predict(self.df[var_explicatives].loc[index_NAN])
            self.df[var].loc[index_NAN] = pred


    def impute_categorical(self):
        nan_table = self.get_nan_table(self.df)
        var_to_impute = nan_table[(self.df['type'] == 'Catégorielle') and (self.df["Pourcentage de valeurs manquantes"] > 0)].columns.to_list()

        for var in var_to_impute:
            self.df[var] = self.df.groupby("Pclass", group_keys=False)[var].apply(lambda x : x.fillna(x.mode()[0]))



    def clean_data(self):
        self.get_type_list()
        self.impute_categorical()
        self.impute_numerical()
        self.remove_outliers()
