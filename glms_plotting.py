
from glms_plotting_utils import *



ROOT_PATH_AXEL = os.path.join(r"/mnt/lsens-analysis/", 'Axel_Bisi', 'NWBFull_bis')
ROOT_PATH_MYRIAM = os.path.join(r"/mnt/lsens-analysis/", 'Myriam_Hamon',
                                'NWBFull')

ROOT_PATH_AXEL = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWBFull_bis')
ROOT_PATH_MYRIAM = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                'NWBFull')



if __name__ == '__main__':

    experimenter = 'Myriam_Hamon'

    info_path = os.path.join(r"/mnt/lsens-analysis/", 'Myriam_Hamon',
                             'mice_info')
    output_path = os.path.join(r"/mnt/lsens-analysis/", experimenter, 'combined_results')

    base_path = r"M:\analysis\Myriam_Hamon\results"
    
    info_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                'mice_info')
    output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'combined_results')

    all_nwb_names = os.listdir(ROOT_PATH_AXEL)
    all_nwb_mice = [name.split('_')[0] for name in all_nwb_names]

    myriam_nwb_names = os.listdir(ROOT_PATH_MYRIAM)
    all_nwb_names.extend(myriam_nwb_names)
    all_nwb_mice.extend([name.split('_')[0] for name in myriam_nwb_names])


    subject_ids = [
         "AB151", "AB157", "AB134", "AB141", "AB120",
        "AB145", "AB147", "AB162", "AB127", "AB125", "AB102",
        "AB154", "AB139", "AB119", "AB153", "AB133", "AB121", "AB156",
        "AB142", "AB140", "AB159", "AB126", "AB122", "AB123", "AB144",
        "AB152", "AB130", "AB117", "AB164", "AB150", "AB138", "AB129", "AB149",
        "AB131", "AB136", "AB163","AB132", "AB094",
        "AB085", "AB104", "AB143", "AB116", "AB128", "AB124"
    ] # bad "AB158""AB107" AB093,"AB092",AB086, AB144, AB149, AB087 "AB095"

    subject_ids = [   'AB093', 'AB140', 'AB159', 'AB126', 'AB122',
        'AB123', 'AB082', 'AB124', 'AB086', 'AB151', 'AB157', 'AB134', 'AB141', 'AB120', 'AB145', 'AB147',
        'AB162', 'AB127', 'AB125',  'AB102', 'AB092', 'AB154', 'AB139', 'AB119', 'AB153',
        'AB080', 'AB133', 'AB121', 'AB156', 'AB142', 'AB131',
        'AB144', 'AB152', 'AB130', 'AB117', 'AB164', 'AB150', 'AB138', 'AB129',
         'AB136', 'AB163', 'AB158', 'AB087', 'AB132', 'AB094', 'AB095', 'AB085', 'AB104',
        'AB143', 'AB116', 'AB128',
        'MH037', 'MH038', 'MH034','MH036', 'MH035', 'MH032', 'MH031', 'MH029',
        'MH026',  'MH015', 'MH017', 'MH018', 'MH019', 'MH020',                                                

    ] # AB107''AB149',
    # subject_ids = [   'AB131', 'AB138', 'AB095', 'AB153'                                     

    # ] # AB107''AB149',
 
    #excluded for now : : ['MH030' 'MH023' 'MH014' 'MH021' 'MH022' 'MH027' 'MH028']
    # subject_ids = ["AB131"]
    # subject_ids = ['AB162']
    # git_version = '40fbc11'
    git_version = '4227ca6'
    git_version = 'b394470'
    git_version = '15127ae'
    # git_version = '74987e2'
    # git_version = '807b8e7'
    git_version = '935b6e1'    
    git_version = 'acbce87'
    # git_version = 'a784830'
    single_mouse = False
    # over_mice = True
    over_mice = True
    compare_gits = False
    git_versions = ['15127ae', 'b394470']
    git_versions = ['b394470', 'b394470'] 
    git_versions = [ 'acbce87' ]

    if single_mouse:

        plots = [ 'average_kernels_by_region']
        # plots = ['metrics'] 'per_unit_kernel_plots',, 'average_predictions_per_trial_types'
        plots = [ 'average_predictions_per_trial_types', 'metrics']
        plots = [ 'average_kernels_by_region', 'average_predictions_per_trial_types', 'metrics']
        plots = ['create_summary', 'metrics']
        plots = [ 'create_summary']
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

            model_path = os.path.join(output_path, subject_id, 'whisker_0', 'unit_glm', 'models')
            if not os.path.exists(model_path):
                print(f"No models found for {subject_id}")
                continue

            mouse_results_path = os.path.join(output_path, subject_id, 'whisker_0', 'unit_glm', git_version)
            if not os.path.exists(mouse_results_path):
                os.makedirs(mouse_results_path)

            mouse_glm_results(nwb_list = nwb_files,model_path = model_path,plots= plots, output_path =mouse_results_path, info_path= info_path, git_version =git_version)

    if over_mice:
        plots = ['metrics']
        plots = ['average_kernels_by_region']
        plots= [ 'kernel_consistency' ]
        # Get list of NWB files for each mouse
        for git_version in git_versions:
            nwb_list = [os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('AB')]
            nwb_list.extend([os.path.join(ROOT_PATH_MYRIAM, name) for name in all_nwb_names if name.startswith('MH')])

            # Keep NWB files for specified subject IDs
            nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in subject_ids)]
            # over_mouse_glm_results(nwb_list= nwb_list,  plots= plots,  info_path = info_path, output_path= output_path, git_version =git_version,
            #                     day_to_analyze=0)
            over_mouse_glm_results_new(subject_ids,nwb_list, plots, output_path, git_version, day_to_analyze = 0)
        
    if compare_gits :
                # Get list of NWB files for each mouse
        nwb_list = [os.path.join(ROOT_PATH_AXEL, name) for name in all_nwb_names if name.startswith('AB')]
        nwb_list.extend([os.path.join(ROOT_PATH_MYRIAM, name) for name in all_nwb_names if name.startswith('MH')])

        # Keep NWB files for specified subject IDs
        nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in subject_ids)]
        plots = ['metrics']
        over_mouse_compare_git_results_new(subject_ids, plots,info_path, output_path, git_versions, day_to_analyze = 0)

        # over_mouse_compare_git_results(nwb_list, plots,info_path, output_path, git_versions, day_to_analyze = 0)
