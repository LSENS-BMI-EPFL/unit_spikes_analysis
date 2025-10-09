import os.path
from idlelib.autocomplete import TRY_A
from importlib.metadata import FastPath

from glms_plotting_utils import *



ROOT_PATH_AXEL = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWBFull')
ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                'NWBFull')



if __name__ == '__main__':

    experimenter = 'Myriam_Hamon'

    info_path = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'z_LSENS', 'Share', 'Axel_Bisi_Share',
                             'dataset_info')
    output_path = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'combined_results')

    base_path = r"M:\analysis\Myriam_Hamon\results"

    all_nwb_names = os.listdir(ROOT_PATH_AXEL)
    all_nwb_mice = [name.split('_')[0] for name in all_nwb_names]

    myriam_nwb_names = os.listdir(ROOT_PATH_MYRIAM)
    all_nwb_names.extend(myriam_nwb_names)
    all_nwb_mice.extend([name.split('_')[0] for name in myriam_nwb_names])


    subject_ids = [ "AB131", "AB138", "AB147", "AB162"]
    # subject_ids = ["AB131"]
    git_version = '1cce900'
    single_mouse = True
    # over_mice = True
    # over_mice = False

    if single_mouse:

        plots = [ 'average_kernels_by_region']
        # plots = ['metrics'] 'per_unit_kernel_plots',, 'average_predictions_per_trial_types'
        plots = [ 'average_predictions_per_trial_types']
        for subject_id in subject_ids:
            print(" ")
            print(f"Subject ID : {subject_id}")

            # Take subset of NWBs
            nwb_names = [name for name in all_nwb_names if subject_id in name]
            if subject_id.startswith('AB'):
                nwb_files = [os.path.join(ROOT_PATH_AXEL, name) for name in nwb_names]
            elif subject_id.startswith('MH'):
                nwb_files = [os.path.join(ROOT_PATH_MYRIAM, name) for name in nwb_names]

            if not nwb_files:
                print(f"No NWB files found for {subject_id}")
                continue

            model_paths = os.path.join(output_path, subject_id, 'whisker_0', 'unit_glm', 'models')
            if not os.path.exists(model_paths):
                print(f"No models found for {subject_id}")
                continue

            mouse_results_path = os.path.join(output_path, subject_id, 'whisker_0', 'unit_glm', git_version)
            if not os.path.exists(mouse_results_path):
                os.makedirs(mouse_results_path)

            mouse_glm_results(nwb_list = nwb_files,model_path = model_paths,plots= plots, output_path =mouse_results_path, git_version =git_version)

    if over_mice:
        plots = [ 'metrics']

        # Get list of NWB files for each mouse
        nwb_list = [os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('AB')]
        nwb_list.extend([os.path.join(ROOT_PATH_MYRIAM, name) for name in all_nwb_names if name.startswith('MH')])

        # Keep NWB files for specified subject IDs
        nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in subject_ids)]
        over_mouse_glm_results(nwb_list= nwb_list,  plots= plots,  info_path = info_path, output_path= output_path, git_version =git_version,
                               day_to_analyze=0)