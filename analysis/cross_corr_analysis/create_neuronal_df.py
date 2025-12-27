import os
import time
import pathlib
import numpy as np
import pandas as pd
import roc_utils as ru 
import allen_utils_old as allen
import NWB_reader_functions as nwb_reader

DATA_PATH = os.path.join('/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Dana_Shayakhmetova', 'NWBFull_Copy')
OUTPUT_DIR = os.path.join('/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Dana_Shayakhmetova', 'dynamic_analysis_dec16')

def calculate_raw_evoked_responses(neural_data):
    """Calculate evoked response as difference between post- and pre-stimulus raw spike counts."""
    neural_data = neural_data.copy()
    neural_data['raw_evoked_response'] = neural_data.apply(
        lambda row: row['raw_post_spikes'] - row['raw_pre_spikes'], axis=1
    )
    return neural_data

def calculate_evoked_responses(neural_data):
    """Calculate evoked response as difference between post- and pre-stimulus firing rates."""
    neural_data = neural_data.copy()
    neural_data['rate_evoked_response'] = neural_data.apply(
        lambda row: row['post_firing_rate'] - row['pre_firing_rate'], axis=1
    )
    return neural_data

def prepare_data(df):
    """Clean and filter neural data: add area labels, convert to arrays, remove excluded areas."""
    df =  allen.create_area_custom_column(df)
    
    #make them arrays
    df['raw_post_spikes'] = df['raw_post_spikes'].apply(np.array)
    df['raw_pre_spikes'] = df['raw_pre_spikes'].apply(np.array)
    df['post_firing_rate'] = df['post_firing_rate'].apply(np.array)
    df['pre_firing_rate'] = df['pre_firing_rate'].apply(np.array)

    #remove columns 
    df = df.loc[:, ~df.columns.str.startswith('ccf')]
    values_to_drop = allen.get_excluded_areas()
    df = df[~df['area_custom'].isin(values_to_drop)]

    df = df.drop(columns=["session_id", "event", "context", "firing_rate", "spike_times"])

    #Filter areas 
    return df

def process_and_combine_spike_data_with_baseline(root_path):
    """
    Extracts spike data, including a 2s baseline, from all valid 
    (whisker, day 0) NWB files in a given folder path.
    """
    try:
        all_nwb_names = os.listdir(os.path.join(root_path))
    except FileNotFoundError:
        print(f"Error: The directory was not found: {root_path}")
        return

    all_nwb_files = [os.path.join(root_path, name) for name in all_nwb_names if name.endswith(".nwb")]

    print(f"Found {len(all_nwb_files)} NWB files to process.")
    all_dfs_list = []

    for i, file_path in enumerate(all_nwb_files):
        print(f"\nProcessing file {i+1}/{len(all_nwb_files)}: {os.path.basename(file_path)}")

        # Validating session 
        try:
            session_info = nwb_reader.get_bhv_type_and_training_day_index(file_path)
            is_valid_file = (session_info[0] == 'whisker' and session_info[1] in [0])
            
        except Exception as e:
            print(f"Could not check session info for {os.path.basename(file_path)}. Skipping. Error: {e}")
            continue 

        start_time = time.time()

        # Extracting data 
        if is_valid_file:
            print("Found a valid whisker day 0 file.")
            try:
                single_mouse_df = ru.extract_spike_data_with_baseline(file_path)
                
                if single_mouse_df is not None and not single_mouse_df.empty:
                    all_dfs_list.append(single_mouse_df)
                    end_time = time.time()
                    print(f"Found {len(single_mouse_df)} entries in {round(end_time-start_time,2)} s")
                else:
                    print(f"No data was returned for {os.path.basename(file_path)}")

            except Exception as e:
                print(f"An error occurred when trying to extract spike data for {file_path}: {e}")
        else:
            print("Not a whisker day 0 file. Skipping.")

    if not all_dfs_list:
        print("\nData resulted in an empty list.")
        return

    combined_df = pd.concat(all_dfs_list, ignore_index=True)
    return combined_df

def main():
    print("\nPulling AB and MH NWB files...")
    data_path = os.path.join(DATA_PATH)
    final_dataframe = process_and_combine_spike_data_with_baseline(data_path)
    if final_dataframe is None or final_dataframe.empty:
        print("Script terminated. No data found or dataframe is empty.")
        return
    print("Done.")

    print("\nPreparing data (adding allen, etc.)...")
    organized_data = prepare_data(final_dataframe)
    print("Done.")


    print("\nCalculating evoked response (raw and firing rate)...")
    organized_data = calculate_evoked_responses(organized_data)
    organized_data = calculate_raw_evoked_responses(organized_data)

    print("\nSaving final dataframe on server...")
    os.makedirs(os.path.join(OUTPUT_DIR, 'neuronal_data'), exist_ok=True)
    organized_data.to_pickle(os.path.join(OUTPUT_DIR,'neuronal_data', 'neuronal_data_mh_ab.pkl'))
    organized_data.to_csv(os.path.join(OUTPUT_DIR, 'neuronal_data', 'neuronal_data_mh_ab.csv'))
    organized_data.to_parquet(
        os.path.join(OUTPUT_DIR, 'neuronal_data','neuronal_data_mh_ab.parquet'),
        engine='pyarrow',    
        index=False,
        compression='snappy')
    

    print("Script done.")



if __name__ == '__main__':
    main()





