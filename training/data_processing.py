import numpy as np

class TranformData:
    def __init__(self, df):
        self.df = df

    def change_column_name(self, new_columns_list):
        self.df.columns = new_columns_list
        return self.df

    def transform_columns_in_num(self, columns):
        self.df[columns] = self.df[columns].apply(lambda x: x * 1000, axis=1)
        return self.df

    def transform_percentage_in_number(self, columns_list, column_times, new_columns_list):
        self.df[columns_list] = self.df[columns_list].apply(lambda x: x.str[:-1]\
                                                .apply(lambda number: \
                                                    float(number) / 100), axis=1)
        
        self.df[new_columns_list] =  self.df[columns_list].apply(lambda x: x * self.df[column_times])
        
        return self.df.drop(columns_list, axis=1)

    def divide_two_columns(self, new_column, first_column, second_column):
        self.df[new_column] = self.df[first_column] / self.df[second_column]
        return self.df

    def turn_percentage_in_decimal(self, column):
        self.df[column] = self.df[column].str[:-1].apply(lambda value: float(value)/100)
        return self.df

    def removing_nan_inf(self, column):
        self.df[column] = self.df.valor_parcelas.replace(np.nan, 0)
        self.df[column] = self.df.valor_parcelas.replace(np.inf, 0)

        return self.df

    def transforming_string_into_category(self, string_column, cat_column_name):
        self.df[cat_column_name] = self.df[string_column].apply(lambda x: 1 if x == "SIM" else 0)
        return self.df.drop(string_column, axis=1)

    def remove_object_column(self):
        lst_columns = list(self.df.select_dtypes(include="object").columns)
        self.df.drop(lst_columns, axis=1, inplace=True)

        return self.df

def train_test_using_year(df, cat_target_column, year_column, last_year, sample = 10):
    X_train = df[df[year_column] < last_year].drop([cat_target_column], axis=1).replace(np.nan, 0)
    y_train = df[df[year_column] < last_year][cat_target_column]

    X_test = df[df[year_column] == last_year].drop([cat_target_column], axis=1)
    y_test = df[df[year_column] == last_year][cat_target_column]

    return X_train, X_test[:sample], y_train,  y_test[:sample]