import h5py
import numpy as np
from rich import print
import h5py
import pandas as pd
import matplotlib.pyplot as plt

def unpack_data(outfile, parameters):
    unpack_model = {}
    with h5py.File(outfile, "r") as f:
        unpack_model["para_w_1"] = f["para_w_1"][()]
        unpack_model["loss_loss"] = f["loss_loss"][()]

        unpack_model["nstates_train"] = f["nstates_train"][()]
        unpack_model["nstates_test"] = f["nstates_test"][()]
        unpack_model["state_ind_for_test"] = f["state_ind_for_test"][()]

        unpack_model["train/loss"] = f["train/"+"loss"][()] / unpack_model["nstates_train"]
        unpack_model["train/sloss"] = f["train/"+"sloss"][()] / unpack_model["nstates_train"]
        unpack_model["train/dloss"] = f["train/"+"dloss"][()] / unpack_model["nstates_train"]
        unpack_model["train/penalty"] = f["train/"+"penalty"][()] / unpack_model["nstates_train"]

        unpack_model["test/loss"] = f["test/"+"loss"][()] / unpack_model["nstates_test"]
        unpack_model["test/sloss"] = f["test/"+"sloss"][()] / unpack_model["nstates_test"]
        unpack_model["test/dloss"] = f["test/"+"dloss"][()] / unpack_model["nstates_test"]
        unpack_model["test/penalty"] = f["test/"+"penalty"][()] / unpack_model["nstates_test"]

        for i, kk in enumerate(parameters):
            unpack_model["rdmd_params/" + kk] = f["train/"+"rdmd_params/" + kk][()]

    return unpack_model


def make_avg_CSV_file(nCV:int, model_name:str, parameters:list, dir:str):

    unpacked_model_files = [] 

    for i in range(nCV):
        outfile = f'{dir}/{model_name}/CV{i}_model_output.hdf5'
        unpacked_model_files.append(unpack_data(outfile, parameters))

    df = pd.DataFrame(unpacked_model_files)
    df.to_csv(f"{dir}/{model_name}_nCV{nCV}.csv", index_label='run') 

    return 

if __name__ == "__main__":
    nCV = 30
    parameters = ["E0", "t", "U"]
    make_avg_CSV_file(nCV, 't-U', parameters, "../CVmodels")

    parameters = ["E0", "t", "U", "ni_hop"]
    make_avg_CSV_file(nCV, 't-U-W', parameters, "../CVmodels")

    parameters = ["E0", "t", "U", "V"]
    make_avg_CSV_file(nCV, 't-U-V', parameters, "../CVmodels")

    parameters = ["E0", "t", "U", "J"]
    make_avg_CSV_file(nCV, 't-U-J', parameters, "../CVmodels")
