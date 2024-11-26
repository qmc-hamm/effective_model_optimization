import h5py
import pandas as pd


def unpack_data(infile: str, parameters: list[str]):
    """ Unpack Optimized Solved Model from *.hdf5 files. Exports the loss data, setting parameters, and model
    parameters.

    Parameters
    ----------
    infile : str
        Optimized model file name (*.hdf5).
    parameters : list[str]
        List of model paramters.

    Returns
    -------
    dict
        dict to be turned into csv file.
    """
    unpack_model = {}
    with h5py.File(infile, "r") as f:
        unpack_model["w0"] = f["para_w_0"][()]
        unpack_model["loss_loss"] = f["loss_loss"][()]

        unpack_model["nstates_train"] = f["nstates_train"][()]
        unpack_model["nstates_test"] = f["nstates_test"][()]
        unpack_model["state_ind_for_test"] = f["state_ind_for_test"][()]

        unpack_model["train/loss"] = f["train/" + "loss"][()] / unpack_model["nstates_train"]
        unpack_model["train/sloss"] = f["train/" + "sloss"][()] / unpack_model["nstates_train"]
        unpack_model["train/dloss"] = f["train/" + "dloss"][()] / unpack_model["nstates_train"]
        unpack_model["train/penalty"] = f["train/" + "penalty"][()] / unpack_model["nstates_train"]

        unpack_model["test/loss"] = f["test/" + "loss"][()] / unpack_model["nstates_test"]
        unpack_model["test/sloss"] = f["test/" + "sloss"][()] / unpack_model["nstates_test"]
        unpack_model["test/dloss"] = f["test/" + "dloss"][()] / unpack_model["nstates_test"]
        unpack_model["test/penalty"] = f["test/" + "penalty"][()] / unpack_model["nstates_test"]

        for i, kk in enumerate(parameters):
            unpack_model["rdmd_params/" + kk] = f["train/" + "rdmd_params/" + kk][()]

    return unpack_model


def unpack_CV_data(infile: str, parameters: list[str]):
    """ Unpack Optimized Solved Model from *.hdf5 files. Exports the loss data, setting parameters, and model
    parameters.

    Parameters
    ----------
    infile : str
        Optimized model file name (*.hdf5).
    parameters : list[str]
        List of model paramters.

    Returns
    -------
    dict
        dict to be turned into csv file.
    """
    unpack_model = {}
    with h5py.File(infile, "r") as f:
        unpack_model["w0"] = f["para_w_0"][()]
        unpack_model["loss_loss"] = f["loss_loss"][()]

        unpack_model["nstates_train"] = f["nstates_train"][()]
        unpack_model["nstates_test"] = f["nstates_test"][()]
        unpack_model["state_ind_for_test"] = f["state_ind_for_test"][()]

        unpack_model["train/loss"] = f["train_loss"][()]
        unpack_model["train/sloss"] = f["train_sloss"][()] / unpack_model["nstates_train"]
        unpack_model["train/dloss"] = f["train_dloss"][()] / unpack_model["nstates_train"]
        unpack_model["train/penalty"] = f["penalty"][()]

        unpack_model["test/loss"] = f["test_loss"][()]
        unpack_model["test/sloss"] = f["test_sloss"][()] / unpack_model["nstates_test"]
        unpack_model["test/dloss"] = f["test_dloss"][()] / unpack_model["nstates_test"]
        unpack_model["test/penalty"] = f["penalty"][()]

        for i, kk in enumerate(parameters):
            unpack_model["rdmd_params/" + kk] = f["rdmd_params/" + kk][()]

    return unpack_model


def make_avg_CSV_file(nCV: int, model_name: str, parameters: list[str], dir: str):
    """_summary_

    Parameters
    ----------
    nCV : int
        Number of CV runs.
    model_name : str
        Name of model in directory structure.
    parameters : list[str]
        List of model paramters.
    dir : str
        Name of director header.
    """

    unpacked_model_files = []

    for i in range(nCV):
        infile = f'{dir}/{model_name}/CV{i}_model_output.hdf5'
        unpacked_model_files.append(unpack_CV_data(infile, parameters))

    df = pd.DataFrame(unpacked_model_files)
    df.to_csv(f"{dir}/{model_name}_nCV{nCV}.csv", index_label='run')

    return


if __name__ == "__main__":
    nCV = 30
    #parameters = ["E0", "t", "U"]
    #make_avg_CSV_file(nCV, 't-U', parameters, "../CVmodels")
    #make_avg_CSV_file(nCV, 't-U', parameters, "../CVmodels_p2")
    #make_avg_CSV_file(nCV, 't-U', parameters, "../CVmodels_p3")
    #make_avg_CSV_file(nCV, 't-U', parameters, "../CVmodels_p4")

    #nCV = 30
    parameters = ["E0", "t", "U", "tdiag"]
    make_avg_CSV_file(nCV, 't-tprime-U', parameters, "../CVmodels_p5")
    quit()

    parameters = ["E0", "t", "U"]
    make_avg_CSV_file(nCV, 't-U', parameters, "../CVmodels_p5")

    parameters = ["E0", "t", "U", "ni_hop"]
    make_avg_CSV_file(nCV, 't-U-W', parameters, "../CVmodels_p5")

    parameters = ["E0", "t", "U", "V"]
    make_avg_CSV_file(nCV, 't-U-V', parameters, "../CVmodels_p5")

    parameters = ["E0", "t", "U", "J"]
    make_avg_CSV_file(nCV, 't-U-J', parameters, "../CVmodels_p5")
