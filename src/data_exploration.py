import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


class DataExploration():
    def __init__(self, df):
        self.df = df

    def nan_table(self):
        nan_table = pd.DataFrame(columns=["Variable", "Pourcentages de valeurs manquantes"])

        for col in self.df:
            pcentage = (self.df[col].isna().sum() / len(self.df)) * 100
            nan_table.loc[len(nan_table)] = [col, pcentage]

        nan_table = nan_table.sort_values(by=["Pourcentages de valeurs manquantes"], ascending=False).reset_index(
            drop=True)

        fig = px.histogram(nan_table,
                           x="Variable",
                           y="Pourcentages de valeurs manquantes",
                           title="Distributions des Valeurs Manquantes")

        fig.update_yaxes(title_text="Pourcentages de Valeurs Manquantes")
        fig.show()

        return (nan_table)
