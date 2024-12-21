import tempfile
from tempfile import tempdir

import h5py
import pandas as pd
import loss_function_function as loss_function
import os
import itertools
import argparse
import sys
import mlflow

from plot_model import plot_model

all_rs = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0] # all ai_data rs

def runCV(named_terms,
          ai_dir,
          model_descriptors,
          nroots,
          onebody_params,
          twobody_params,
          train_rs,
          test_rs,
          minimum_1s_occupation=3.7,
          w0=1,
          beta=0,
          p=0,
          guess_params=None,
          state_cutoff=None,
          niter_opt=1,
          tol_opt=1.0,
          maxfev_opt=1.0,
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
        ai_df = pd.read_csv(ai_dir + f"r{r}.csv")
        ai_df = ai_df[ai_df.E0 > minimum_1s_occupation]
        ai_df = ai_df[ai_df.U < 1.3]  # Need to remove the top two states from the optimization
        if state_cutoff is not None:
            ai_df = ai_df[ai_df.state < state_cutoff]
        ai_df = ai_df.reset_index()
        ai_df_rs[f'r{r}'] = ai_df

    #print(ai_df_rs)

    matches = onebody_params + twobody_params
    weights = [w0, 1 - w0]

    loss_function.mapping(
        onebody,
        twobody,
        onebody_params,
        twobody_params,
        ai_df_rs,
        nroots,
        model_descriptors,
        matches,
        train_rs,
        test_rs,
        weights,
        beta,
        p,
        guess_params,
        niter_opt = niter_opt,
        tol_opt = tol_opt,
        maxfev_opt = maxfev_opt,
    )


def make_name(parameters):
    return "_".join(parameters[0]) + "_" + "_".join(parameters[1])

def main(parameters, state_cutoff, w0, train_rs, niter_opt, tol_opt, maxfev_opt, nCV_iter):
        test_rs = list( set(all_rs) - set(train_rs))#.sort()
        test_rs.sort()
        print("Test rs values:", test_rs)
        with mlflow.start_run():
            # Write model and plots to temp dir
            with tempfile.TemporaryDirectory() as output_dir:
                model_files = []
                for i in range(nCV_iter):
                    pname = make_name(parameters)
                    dirname = os.path.join(output_dir,f"func_model_data_{state_cutoff}_{w0}")
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    model_file_path = f"{dirname}/{pname}_{i}.hdf5"

                    runCV(named_terms="symmetric_operators.hdf5",
                          ai_dir="ai_data/",
                          model_descriptors=model_file_path,
                          nroots=36,
                          onebody_params=parameters[0],
                          twobody_params=parameters[1],
                          train_rs=train_rs,
                          test_rs=test_rs,
                          w0=w0,
                          beta=0,
                          p=1,
                          state_cutoff=state_cutoff,
                          niter_opt=niter_opt,
                          tol_opt=tol_opt,
                          maxfev_opt=maxfev_opt
                          )
                    model_files.append(model_file_path)
                plot_model(output_dir, model_files, parameters) # Artifacts the plots to mlflow inside function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=str, nargs="+")
    parser.add_argument("--train_rs", type=str)
    parser.add_argument("--state_cutoff", type=int)
    parser.add_argument("--w0", type=float)
    parser.add_argument("--niter_opt", type=int)
    parser.add_argument("--tol_opt", type=float)
    parser.add_argument("--nCV_iter", type=int, default=1)
    parser.add_argument("--maxfev_opt", type=int, default=1)
    args = parser.parse_args()
    parameters = (args.parameters[0].split(','), args.parameters[1].split(','))
    state_cutoff = args.state_cutoff
    w0 = args.w0
    train_rs = [float(r) for r in args.train_rs.split(",")]
    niter_opt = args.niter_opt
    tol_opt = args.tol_opt
    nCV_iter = args.nCV_iter
    maxfev_opt = args.maxfev_opt

    main(parameters, state_cutoff, w0, train_rs, niter_opt, tol_opt, maxfev_opt, nCV_iter)
