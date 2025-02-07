import numpy as np
import pandas as pd
import tempfile
import h5py
import os
import argparse
import mlflow

import loss_function_function as loss_function
from plot_model import plot_model

all_rs = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4, 5.0] # all ai_data rs


def runCV(named_terms,
          ai_dir,
          model_descriptors,
          nroots,
          onebody_params,
          twobody_params,
          train_rs,
          param_functions,
          minimum_1s_occupation=3.7,
          w0=1,
          p=1,
          guess_params=None,
          state_cutoff=None,
          niter_opt=2,
          tol_opt=1e-2,
          maxfev_opt=1000.0,
          tmpdirname=None
          ):

    onebody = {}
    twobody = {}
    onebody_keys = []
    twobody_keys = []
    with h5py.File(named_terms, "r") as f:
        for k in f["onebody"].keys():
            onebody[k] = f[f"onebody/{k}"][()]
            onebody_keys.append(k)
        for k in f["twobody"].keys():
            twobody[k] = f[f"twobody/{k}"][:]
            twobody_keys.append(k)

    ai_df_rs = {}
    for r in all_rs:
        ai_df = pd.read_csv(ai_dir)
        ai_df = ai_df[ai_df.r == r]
        ai_df = ai_df[ai_df.doccp < 0.8]  # Only doing spin states for now
        if state_cutoff is not None:
            ai_df = ai_df[ai_df.state < state_cutoff]
        ai_df = ai_df.reset_index()
        ai_df_rs[f'r{r}'] = ai_df

    #print(ai_df_rs)

    matches = onebody_params + twobody_params
    weights = [w0, 1 - w0]

    loss_function.setup_train(
        onebody,
        twobody,
        onebody_params,
        twobody_params,
        ai_df_rs,
        nroots,
        model_descriptors,
        matches,
        train_rs,
        param_functions,
        weights,
        p,
        guess_params,
        niter_opt=niter_opt,
        tol_opt=tol_opt,
        maxfev_opt=maxfev_opt,
    )


def runInference(named_terms,
                 ai_dir,
                 model_descriptors,
                 inference_name,
                 nroots,
                 onebody_params,
                 twobody_params,
                 rs,
                 minimum_1s_occupation=3.7,
                 state_cutoff=None,
                 tmpdirname=None
                 ):

    onebody = {}
    twobody = {}
    onebody_keys = []
    twobody_keys = []
    with h5py.File(named_terms, "r") as f:
        for k in f["onebody"].keys():
            onebody[k] = f[f"onebody/{k}"][()]
            onebody_keys.append(k)
        for k in f["twobody"].keys():
            twobody[k] = f[f"twobody/{k}"][:]
            twobody_keys.append(k)

    ai_df_rs = {}
    for r in rs:
        ai_df = pd.read_csv(ai_dir)
        ai_df = ai_df[ai_df.r == r]
        ai_df = ai_df[ai_df.doccp < 0.8]  # Only doing spin states for now
        if state_cutoff is not None:
            ai_df = ai_df[ai_df.state < state_cutoff]
        ai_df = ai_df.reset_index()
        ai_df_rs[f'r{r}'] = ai_df

    matches = onebody_params + twobody_params

    params_dict = {}

    with h5py.File(model_descriptors, 'r') as f:
        for r in rs:
            param_list = []
            for parameter in matches:
                param_list.append(f[f'r{r}/rdmd_params/{parameter}'][()])
            params_dict[f'r{r}'] = np.array(param_list)

    loss_function.inference(
        onebody,
        twobody,
        onebody_params,
        twobody_params,
        ai_df_rs,
        inference_name,
        nroots,
        model_descriptors,
        matches,
        rs,
        params_dict,
    )


def make_name(parameters):
    return "_".join(parameters[0]) + "_" + "_".join(parameters[1])


def main(parameters, state_cutoff, w0, train_rs, niter_opt, tol_opt, maxfev_opt, nCV_iter, param_functions):
    with mlflow.start_run():
        # Write model and plots to temp dir
        with tempfile.TemporaryDirectory() as output_dir:
            model_files = []
            for i in range(nCV_iter):
                pname = make_name(parameters)
                dirname = os.path.join(output_dir, f"func_model_data_{state_cutoff}_{w0}")
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                model_file_path = f"{dirname}/{pname}_{i}.hdf5"

                runCV(named_terms="hchain4_named_operators.hdf5",
                      ai_dir="ai_data/hchain4.csv",
                      model_descriptors=model_file_path,
                      nroots=36,
                      onebody_params=parameters[0],
                      twobody_params=parameters[1],
                      train_rs=train_rs,
                      param_functions=param_functions,
                      w0=w0,
                      p=1,
                      state_cutoff=state_cutoff,
                      niter_opt=niter_opt,
                      tol_opt=tol_opt,
                      maxfev_opt=maxfev_opt
                      )

                runInference(named_terms="hchain6_named_operators.hdf5",
                             ai_dir="ai_data/hchain6.csv",
                             model_descriptors=model_file_path,
                             inference_name="natoms6_casscf",
                             nroots=100,
                             onebody_params=parameters[0],
                             twobody_params=parameters[1],
                             rs=train_rs,
                             state_cutoff=state_cutoff,
                             )

                runInference(named_terms="hchain8_named_operators.hdf5",
                             ai_dir="ai_data/hchain8_casscf.csv",
                             model_descriptors=model_file_path,
                             inference_name="natoms8_casscf",
                             nroots=200,
                             onebody_params=parameters[0],
                             twobody_params=parameters[1],
                             rs=[3.0],#[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4],
                             state_cutoff=state_cutoff,
                             )

                runInference(named_terms="hchain8_named_operators.hdf5",
                             ai_dir="ai_data/hchain8_vmc.csv",
                             model_descriptors=model_file_path,
                             inference_name="natoms8_vmc",
                             nroots=200,
                             onebody_params=parameters[0],
                             twobody_params=parameters[1],
                             rs=[3.0],#[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4],
                             state_cutoff=state_cutoff,
                             ) 

                mlflow.log_artifact(model_file_path)
                model_files.append(model_file_path)
            plot_model(output_dir, model_files, ["natoms6_casscf", "natoms8_casscf", "natoms8_vmc"], parameters) # Artifacts the plots to mlflow inside function


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=str, nargs="+")
    parser.add_argument("--train_rs", type=str)
    parser.add_argument("--state_cutoff", type=int)
    parser.add_argument("--w0", type=float)
    parser.add_argument("--niter_opt", type=int)
    parser.add_argument("--tol_opt", type=float)
    parser.add_argument("--nCV_iter", type=int, default=1)
    parser.add_argument("--maxfev_opt", type=int, default=1000)
    parser.add_argument("--parameter_functions", type=str, nargs="+")

    args = parser.parse_args()
    parameters = (args.parameters[0].split(','), args.parameters[1].split(','))
    state_cutoff = args.state_cutoff
    w0 = args.w0
    train_rs = [float(r) for r in args.train_rs.split(",")]
    niter_opt = args.niter_opt
    tol_opt = args.tol_opt
    nCV_iter = args.nCV_iter
    maxfev_opt = args.maxfev_opt
    param_functions = args.parameter_functions[0].split(',') + args.parameter_functions[1].split(',')

    main(parameters, state_cutoff, w0, train_rs, niter_opt, tol_opt, maxfev_opt, nCV_iter, param_functions)
    #main((['trace', 't_1'], ['doccp']), None, 0.9, [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4, 5.0], 1, 1, 10, 1, ['independent', 'independent', 'independent'])
