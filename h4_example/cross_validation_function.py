import h5py
import pandas as pd
import loss_function_function as loss_function
import os
import itertools
import argparse
import sys
import mlflow

from plot_model import plot_model


def runCV(named_terms,
          ai_dir,
          model_descriptors,
          nroots,
          onebody_params,
          twobody_params,
          rs,
          minimum_1s_occupation=3.7,
          w0=1,
          beta=0,
          p=0,
          guess_params=None,
          state_cutoff=None,
          niter_opt=1,
          tol_opt=1.0,
          maxfev_opt=1.0
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
        rs,
        weights,
        beta,
        p,
        guess_params,
        niter_opt = niter_opt,
        tol_opt = tol_opt,
        maxfev_opt = 1, # little optimization set to test workflow
    )


def make_name(parameters):
    return "_".join(parameters[0]) + "_" + "_".join(parameters[1])

def main(parameters, state_cutoff, w0, rs, niter_opt, tol_opt, maxfev_opt, nCV_iter):
    with mlflow.start_run():
        for i in range(nCV_iter):
            pname = make_name(parameters)
            dirname = f"func_model_data_{state_cutoff}_{w0}"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            model_file_path = f"{dirname}/{pname}_{i}.hdf5"

            runCV(named_terms="symmetric_operators.hdf5",
                  ai_dir="ai_data/",
                  model_descriptors=model_file_path,
                  nroots=36,
                  onebody_params=parameters[0],
                  twobody_params=parameters[1],
                  rs=rs,
                  w0=w0,
                  beta=0,
                  p=1,
                  state_cutoff=state_cutoff,
                  niter_opt=niter_opt,
                  tol_opt=tol_opt,
                  maxfev_opt=maxfev_opt
                  )
            mlflow.log_artifact(model_file_path)
            plot = plot_model(model_file_path, parameters)
            mlflow.log_artifact(plot)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--parameters", type=str, nargs="+")
        parser.add_argument("--rs", type=str)
        parser.add_argument("--state_cutoff", type=int)
        parser.add_argument("--w0", type=float)
        parser.add_argument("--niter_opt", type=int)
        parser.add_argument("--tol_opt", type=float)
        parser.add_argument("--nCV_iter", type=int, default=1)
        parser.add_argument("--maxfev_opt", type=int, default=1)
        args = parser.parse_args()
        parameters = (args.parameters[0].split(','), args.parameters[1].split(','))
        print(f"parameters: {parameters}")
        sys.exit(0)
        state_cutoff = args.state_cutoff
        w0 = args.w0
        rs = [float(r) for r in args.rs.split(",")]
        niter_opt = args.niter_opt
        tol_opt = args.tol_opt
        nCV_iter = args.nCV_iter
        maxfev_opt = args.maxfev_opt
        main(parameters, state_cutoff, w0, rs, niter_opt, tol_opt, maxfev_opt, nCV_iter)
    else:
        # Hyperparameters
        parameter_sets = [
            (['E0', 't'], ['U']),
            #(['E0', 't', 'tdiag'], ['U']),
            #(['E0', 't'], ['U', 'V']),
            #(['E0', 't'], ['U', 'J']),
            #(['E0', 't'], ['U', 'V', 'J']),
            #(['E0', 't'], ['J']),
            #(['E0'], ['U', 'J']),
            #(['E0', 't','tdiag'], ['U', 'J']),
            #(['E0', 't','tdiag'], ['U', 'V']),
            #(['E0', 't','tdiag'], ['U', 'V','J']),
        ]
        rs_set = [
            #[2.2, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.0, 6.0, 7.0]
            [2.2, 2.8, 3.2, 3.6, 4.0, 4.4] # Test Workflow
        ]
        state_cutoffs = [
            10 # Test Workflow
            # 6, 8, 10, 12, 14
        ]
        w0s = [
            # 1.0, 0.95, 0.9, 0.85, 0.8
            # 1.0, 0.9, 0.8, 0.7, 0.6
            0.9, 0.8 # Test Workflow
        ]

        # Hyperparameter sweep step
        for parameters, rs, state_cutoff, w0 in itertools.product(parameter_sets,
                                                                rs_set,
                                                                state_cutoffs,
                                                                w0s):
                                                                    main(parameters, state_cutoff, w0, rs)
