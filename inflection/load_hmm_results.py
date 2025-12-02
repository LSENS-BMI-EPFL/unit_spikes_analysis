# Imports
import os
import pandas as pd
import pathlib


ROOT_PATH_AXEL = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi\combined_results')
ROOT_PATH_MYRIAM = pathlib.Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Myriam_Hamon\results')


def load_mouse_hmm_results(mouse_id):
    print('Loading learning curve data...')
    data = []
    path_to_data = ROOT_PATH_AXEL
    for m in mouse_id:
        try:
            file_name = f'{m}_whisker_0_whisker_trial_learning_curve_interp.h5'
            path_to_file = os.path.join(path_to_data, m, 'whisker_0', 'learning_curve', file_name)
            df_w = pd.read_hdf(path_to_file)
            data.append(df_w)
        except FileNotFoundError as err:
            print('No whisker curve for:', m)

    data_df = pd.concat(data).reset_index(drop=True)
    return data_df