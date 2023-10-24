import numpy as np
import pandas as pd
from IPython.display import display

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class DataAnalysis():
    def __init__(self, df):
        self.df = df

    def get_numericals_categoricals(self):
        col_categoricals = self.df.select_dtypes(include=["object"]).columns.to_list()
        col_numericals = self.df.select_dtypes(exclude=["object"]).columns.to_list()

        self.col_numericals = col_numericals
        self.col_categoricals = col_categoricals

        return (col_numericals, col_categoricals)

    def transform_var_categorical_in_numerical(self, var_to_num: list):
        for element in self.df.columns:
            self.df[element] = self.df[element].apply(lambda x: np.nan if x == ' ' else x)

        for element in var_to_num:
            self.df[element] = self.df[element].astype("float")

    def transform_var_numerical_in_categorical(self, var_to_cat: list):
        for element in var_to_cat:
            self.df[element] = self.df[element].astype("object")

    def plot_marginal_distributions(self, liste_variable):
        ncols = 2
        nrows = len(liste_variable) // ncols + (len(liste_variable) % ncols > 0)

        plt.figure(figsize=(17, 14))
        plt.subplots_adjust(hspace=1)
        plt.suptitle("Distribution des \nvariables", fontsize=18, y=0.95)

        for n, variable in enumerate(liste_variable):
            ax = plt.subplot(nrows, ncols, n + 1)

            sns.kdeplot(ax=ax, data=self.df[variable], legend=None, shade=True)
            ax.grid(which='major', axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.set_title(variable, loc='center', fontsize=12)

    def show_correlation_matrix(self):
        matrice_corr = self.df[self.col_numericals].corr()
        fig = px.imshow(matrice_corr,
                        x=matrice_corr.columns,
                        y=matrice_corr.columns,
                        color_continuous_scale='YlGnBu',
                        title='Matrice de Corrélation')

        fig.update_xaxes(side='top')
        fig.update_layout(width=800, height=600)
        fig.show()

    def display_missing_values(self):
        missing_values = pd.DataFrame(columns=["Colonne", "Nombre", "Pourcentage de Valeurs Manquantes"])

        for col in self.df.columns:
            nbr_missing = self.df[col].isna().sum()
            pcentage = nbr_missing / len(self.df)
            missing_values.loc[len(missing_values)] = [col, nbr_missing, pcentage * 100]

        missing_values.sort_values(by=["Pourcentage de Valeurs Manquantes"], inplace=True)
        missing_values.reset_index(inplace=True, drop=True)

        return (missing_values)

    def mean_imputation(self):
        col_to_display = []
        for col in self.col_numericals:
            self.df[f'{col}_mean_filled'] = self.df[col].fillna(self.df[col].mean())
            col_to_display.extend([col, f'{col}_mean_filled'])

        display(self.df[col_to_display])

    def mode_imputation(self):
        col_to_display = []
        for col in self.col_categoricals:
            self.df[f'{col}_mode_filled'] = self.df[col].fillna(self.df[col].mode().iloc[0])
            col_to_display.extend([col, f'{col}_mode_filled'])

        display(self.df[col_to_display])

    def Regression_imputation(self):
        col_to_display = []
        for var in self.col_numericals:
            index_NAN = self.df[self.df[var].isna()].index
            var_explicatives = []

            for colonne in self.col_numericals:
                if self.df[colonne].loc[index_NAN].isna().sum() == 0 and var != colonne:
                    var_explicatives.append(colonne)

            df_train = self.df.dropna(how='any')

            reg = LinearRegression().fit(df_train[var_explicatives], df_train[var])

            pred = reg.predict(self.df[var_explicatives].loc[index_NAN])
            self.df[f"{var}_reg_filled"] = self.df[var]
            self.df[f"{var}_reg_filled"].loc[index_NAN] = pred
            col_to_display.extend([var, f'{var}_reg_filled'])

        display(self.df[col_to_display])

    def is_outlier(self):
        col_to_display = []
        for column in self.col_numericals :

            # 1er Quartile
            Q1 = self.df[column].quantile(0.25)

            # 3ème Quartile
            Q3 = self.df[column].quantile(0.75)

            # Inter-Quartile Range (IQR)
            IQR = Q3 - Q1

            # limites, basse & haute
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR

            # Remplace les données inférieur et supérieur à la limite par 1 et les autres par 0
            series = (self.df[column] < limite_inf) | (self.df[column] > limite_sup)
            series = series.astype(int)
            self.df[f'is_outlier_{column}'] = series
            col_to_display.append(f'is_outlier_{column}')

        display(self.df[col_to_display])

    def display_outliers(self):
        outliers_df = pd.DataFrame(columns=["Colonne", "Nombre", "Pourcentage d'outliers"])
        outliers_cols = [x for x in self.df.columns if 'is_outlier' in x]

        for col in outliers_cols:
            nbr_outliers = self.df[col].sum()
            pcentage = nbr_outliers / len(self.df)
            outliers_df.loc[len(outliers_df)] = [col, nbr_outliers, pcentage * 100]

        outliers_df.sort_values(by=["Pourcentage d'outliers"], inplace=True, ascending=False)
        outliers_df.reset_index(inplace=True, drop=True)

        return (outliers_df)

    def impute_train_test_numerical(self):
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
        self.train['Electric range (km)'].fillna(0, inplace=True)
        self.test['Electric range (km)'].fillna(0, inplace=True)

        # Imputation of Fuel Consumption
        var_explicatives = self.get_variable_correlation("Fuel consumption ").index.to_list()
        var_explicatives.remove("Ewltp (g/km)")
        var_explicatives = var_explicatives[:3]

        var_full = var_explicatives.copy()
        var_full.append('Fuel consumption ')

        df_train = self.train[var_full].dropna(how='any')
        index_NAN_train = self.train[self.train["Fuel consumption "].isna()].index
        index_NAN_test = self.test[self.test["Fuel consumption "].isna()].index

        reg = LinearRegression().fit(df_train[var_explicatives], df_train["Fuel consumption "])

        pred_train = reg.predict(self.train.loc[index_NAN_train, var_explicatives])
        pred_test = reg.predict(self.test.loc[index_NAN_test, var_explicatives])

        self.train.loc[index_NAN_train, "Fuel consumption "] = pred_train
        self.test.loc[index_NAN_test, "Fuel consumption "] = pred_test

        self.train.loc[self.train['Fuel consumption '] <= 0, 'Fuel consumption '] = 0
        self.test.loc[self.test['Fuel consumption '] <= 0, 'Fuel consumption '] = 0

        print("Valeurs manquantes numériques imputées ✅")

    def categorical_imputation(self):

        #drop de It car trop de nan ------------------------------------------------------------------------------------
        self.df.drop(columns = ['IT'], inplace = True)

        # Imputation 'VFN' avec 'Cn' -----------------------------------------------------------------------------------
        mode_VFN = self.df.groupby('Cn')['VFN'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        self.df['VFN'] = self.df['VFN'].fillna(self.df['Cn'].map(mode_VFN))

        # Impute 'VFN' with 'Va' if 'Cn' is missing
        self.df['VFN'] = self.df.apply(lambda row: row['Va'] if pd.isna(row['VFN']) and not pd.isna(row['Cn']) else row['VFN'],
                             axis=1)

        # Impute 'VFN' with mode of 'VFN' if both 'Cn' and 'Va' are missing
        mode_VFN = self.df['VFN'].mode().iloc[0]
        self.df['VFN'].fillna(mode_VFN, inplace=True)

        # Imputation de Mp ---------------------------------------------------------------------------------------------
        for col in ['Mk', 'Mh', 'Man', 'Cn', 'Tan']:
            mode_by_group = self.df.groupby(col)['Mp'].transform(
                lambda x: x.mode().iloc[0] if not x.mode().empty else None)

            # Remplace les valeurs manquantes par le mode correspondant
            self.df['Mp'].fillna(mode_by_group, inplace=True)
        # Remplace les valeurs manquantes restantes dans 'Mp' par le mode global de 'Mp'
        global_mode = self.df['Mp'].mode().iloc[0]
        self.df['Mp'].fillna(global_mode, inplace=True)
