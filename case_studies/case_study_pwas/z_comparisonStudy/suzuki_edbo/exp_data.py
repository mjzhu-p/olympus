import pandas as pd
import numpy as np

def get_exp_yield(x_complete,rxn_name= 'suzuki'):

    """
    TODO: add comments
    """

    x = np.round(x_complete,decimals=1)

    # Load the matrix Excel file into a DataFrame
    if rxn_name == 'suzuki':
        matrix_excel_file = 'suzuki_experiment_index_formatted.xlsx'
        sheet_name = 'ohe'
    elif rxn_name == 'direct_arylation':
        matrix_excel_file = 'directarylation_experiment_index_formatted.xlsx'
        # sheet_name = 'ohe'
        sheet_name = 'Sheet3'

    matrix_df = pd.read_excel(matrix_excel_file, sheet_name = sheet_name)
    matching_yield_values = []

    # Iterate through the matrix rows and check for a match
    for index, row in matrix_df.iterrows():
        row_array = np.array(row[1:-1])  # Exclude the 'entry',  and 'yield' columns
        if np.array_equal(row_array, x):
            matching_yield_values = row['yield']
            break

    return -matching_yield_values

