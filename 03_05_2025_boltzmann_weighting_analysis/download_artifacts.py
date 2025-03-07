import os 
from mlflow.client import MlflowClient
from mlflow.entities import ViewType

# In terminal make sure to export MLFlow tracking link: 
# $ export MLFLOW_TRACKING_URI=https://qmc-hamm.ml.software.ncsa.illinois.edu/
client = MlflowClient() 

"""Search through all runs under a parent id, needs work"""
parent_name = "learned-rat-33"
#parent_id = client.get_experiment_by_name(parent_name).info.run_id#.experiment_id # Returns NoneType?
parent_id = "ce57aba59924483895f0f945b08cca35"
parent_run = client.get_run(parent_id)
#experiment_runs = client.search_runs(experiment_ids=[experiment_id])
#experiment_runs = client.search_runs(filter_string=f"tags.mlflow.runName ILIKE '%{parent_name}%'", run_view_type=ViewType.ACTIVE_ONLY)
#experiment_runs = client.search_runs(filter_string=f"tags.mlflow.parentRunId='{parent_id}'")
#tags={MLFLOW_PARENT_RUN_ID: f"{current_active_run_id}"}
#print(help(client.search_runs))
#print("Experiment Runs: ",experiment_runs)

run_ids = ['120ef34cf0cd459699b17389a0140f11',
            '35c7d6d23e66488eb2febfa0e31a4a03',
            'd6ada29a763d4cfe89a23a2dcead3867',
            'b1c990175acb477ca2db21bb6a4af14a',
            '7857cd1381994595a6c1fa5b98d22fdb',
            '4cf5951a23c1477ba21a849ade284a5e',
            '696355b97b3640808e452f6183be6579',
            '31f06a465a8e4c5ba94031ef95fa292f',
            'dbcb71f4b578428ebaed388f091acfea',
            'c90de7cab5244305ad23f2a2303e287f',
            '6caac94630f44fe1b9b2f96e35ece050',
            'feab579bce824445b95498a009a973b8',
            'c72a221b836b4f91bb406fcf16feefe1',
            'e5d16251184c4e63ba06ff1e0b4fdbc0',
            'fa496b3272ba4084a8cebf8b2a45fee2',
            '1e542f030eac4c5eb3e50b4817d90462',
            'ba8bd4fc9bdf4570b374072a30745ed1',
            '645014564b42444b84be53ff72da229d',
            '532baab9a9dd4da5b7c131cb94b02621',
            'f6b8894ba82547e78b68dd5a7302062c',]

for run_id in run_ids:
    #run_id = "f6b8894ba82547e78b68dd5a7302062c"
    run_data_dict = client.get_run(run_id).data.to_dictionary() # Children (like params) of a run
    #print(run_data_dict['params'])

    # search_all_experiments=True

    w0 = run_data_dict['params']['w0']
    temp = run_data_dict['params']['temp']

    dir = "runs"
    if not os.path.exists(dir):
        os.makedirs(dir)
    local_dir = f"{dir}/w0{w0}_temp{temp}/" # where you want to save the data
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    local_path = client.download_artifacts(run_id, "Processed_CV_data.csv", local_dir)
    print("Artifacts downloaded in: {}".format(local_path))