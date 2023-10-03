#data_preprocessing/data_preprocessor.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:

    """
    A class for preprocessing data, including tasks like data reading, imputing missing values,
    removing duplicates and outliers, and applying feature scaling.

    Attributes
    ----------
    -data_name : str
    The name or path to the CSV data file.

    Examples
    --------
    Create an instance of the `DataPreprocessor` class:

    my_instance = DataPreprocessor("data.csv")
    """

    def __init__(self, data_name: str):

        """
        Initializes an instance of the class with a data name.

        Parameters
        ----------
        -data_name (str): The name or path to the CSV data file.

        Example
        my_instance = MyClass("data.csv")
        """
        self.data = data_name

    def get_data(self) -> pd.DataFrame:

        """
        Reads a CSV data file specified during initialization and returns the data as a Pandas DataFrame.

        Returns
        data (pd.DataFrame): The data from the CSV file as a Pandas DataFrame.

        Example:
        my_instance = MyClass("data.csv")
        data = my_instance.get_data()
        """

        data = pd.read_csv(self.data)
        return data

    @staticmethod
    def impute_missing_values(imputer: SimpleImputer, data: pd.DataFrame) -> pd.DataFrame:

        """
        Impute missing values (Most common strategy is mean)

        Parameters
        ----------
        - imputer : SimpleImputer
        The imputer
        - data: pd.DataFrame
        The dataframe

        Returns
        -------
        pd.DataFrame
        The dataframe updated with null values imputed
        """

        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    @staticmethod
    def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:

        """
        Remove duplicate rows from the data.

        Parameters
        --------
        -data: pd.DataFrame

        Returns
        -------
        pd.DataFrame:
        The dataframe with duplicate rows removed.

        Notes:
        This method modifies the dataframe in-place and also returns the modified dataframe.
        """

        return data.drop_duplicates()

    @staticmethod
    def remove_outliers(column_name: str, data: pd.DataFrame) -> pd.DataFrame:

        """
        Remove outliers  from the data.

        Parameters
        ----------
        - column_name : str
        - data : pd.DataFrame
        The column we want to apply the changes

        Returns
        -------
        pd.DataFrame:
        The dataframe with outlier rows removed.

        """

        if column_name == 'medv':
            data = data[~(data[column_name] == 50)]
        elif column_name == 'tax':
            TAX_10: float= data[(data[column_name] < 600) & (data['lstat'] >= 0) & (data['lstat'] < 10)][column_name].mean()
            TAX_20: float = data[(data[column_name] < 600) & (data['lstat'] >= 10) & (data['lstat'] < 20)][column_name].mean()
            TAX_30: float = data[(data[column_name] < 600) & (data['lstat'] >= 20) & (data['lstat'] < 30)][column_name].mean()
            TAX_40: float = data[(data[column_name] < 600) & (data['lstat'] >= 30)][column_name].mean()

            indexes = list(data.index)
            for i in indexes:
                if data[column_name][i] > 600:
                    if (0 <= data['lstat'][i] < 10):
                        data.at[i, column_name] = TAX_10
                    elif (10 <= data['lstat'][i] < 20):
                        data.at[i, column_name] = TAX_20
                    elif (20 <= data['lstat'][i] < 30):
                        data.at[i, column_name] = TAX_30
                    elif (data['lstat'][i] > 30):
                        data.at[i, column_name] = TAX_40
        else:
            return 'Error'

        return data

    @staticmethod
    def feature_scaler(scaler: StandardScaler, data: pd.DataFrame)-> pd.DataFrame:

        """
        Apply feature scaling on the features.

        Parameters
        ----------
        - scaler : StandardScaler
        The scaler from scikit-learn library
        - data : pd.DataFrame

        Returns
        -------
        pd.DataFrame:
        The dataframe with its features scaled.
        """

        return pd.DataFrame(scaler.fit_transform(data.iloc[:, :-1]), columns = data.columns[:-1])
