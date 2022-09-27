import numpy as np
import pandas as pd


# obtain the corresponding data frames (df) from a given (index_list)

def get_student_data(index_list, df):
    test_df = pd.DataFrame()
    for i in index_list:
        df_temp = df[df['StudentID'] == i]
        test_df = pd.concat([test_df, df_temp])
    return test_df

# round to nearest Swiss Grade
def round_to_nearest_grade(x):
    return round(x * 4) / 4

def remove_entries(df, x):
    prep_df = pd.DataFrame()
    index = df['StudentID'].unique()
    for i in index:
        df_temp = df[df['StudentID'] == i].copy()
        df_temp.iloc[x:,[2]] = np.NaN
        prep_df = pd.concat([prep_df, df_temp])
    return prep_df

