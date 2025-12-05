import numpy as np
import pandas as pd
import tempfile
import h5py
import os
import argparse
import mlflow

import loss_function_function as loss_function
from plot_model import plot_model
from plot_thermo import plot_thermo

all_rs = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4, 4.8, 5.0] # all ai_data rs


def runCV(named_terms: str,
          ai_dir: str,
          model_descriptors: str,
          nroots: int,
          onebody_params: list[str],
          twobody_params: list[str],
          train_rs: list[float],
          param_functions: list[str],
          minimum_1s_occupation: float = 0.91,
          w: float = 0.0,
          beta: float = 0.0,
          p: int = 1,
          guess_params: list[float] = None,
          state_cutoff: int = None,
          lamb: float = 1.0,
          niter_opt: int = 1000,
          tol_opt: float = 1e-7,
          maxfev_opt: int = 10000,
          tmpdirname=None
          ):
    """ Wrapper function that optimizes an effective model using loss_function.py.

    Parameters
    ----------
    named_terms : str
        File that holds the onebody and twobody terms.
    ai_dir : str
        File that holds the ab initio data. The descriptor information of the ab initio data. The keys should match model descriptors
    model_descriptors : str
        Name of file to store model in hdf5 format.
    nroots : int
        Number of roots to solve the effective model for.
    onebody_params :list[str]
        List of the key names for the onebody parameters.
    twobody_params : list[str]
        List of the key names for the twobody parameters.
    train_rs : list[float]
        List of the atomic spacing, r, to optimize over.
    param_functions : list[str]
        The function type of the parameter vs r function.
    minimum_1s_occupation : float, optional
        The minimum occupancy per site in the target space, by default 0.91.
    w : float, optional
        The ratio of descriptor to spectrum loss value, by default 0.0.
    beta : float, optional
        Boltmann beta for boltmann weighting term, by default 0.0.
    p : int, optional
        Number of states to leave out as validation, by default 1.
    guess_params : list[float], optional
        Guess parameter values to use for optimizition. If None uses DMD guess values instead. By default None.
    state_cutoff : int, optional
        The state number of ab initio data that is the cutoff value, by default None.
    lamb : float, optional
        Strength of the intruder state penalty term, by default 1.0.
    niter_opt : int, optional
        Number of optimization iterations, by default 2. Passed into scipy.optimize.
    tol_opt : float, optional
        Tolerance of optimization, by default 1e-2. Passed into scipy.optimize.
    maxfev_opt : int, optional
        Maximum function call by optimization procedure, by default 1000.0. Passed into scipy.optimize.
    tmpdirname : _type_, optional
        _description_, by default None

    Returns
    -------
    float
        Loss function dictionary.
    """

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
        ai_df = ai_df[ai_df.delta == 0.0]
        ai_df = ai_df[ai_df.trace >= len(onebody['trace'])*minimum_1s_occupation]
        if state_cutoff is not None:
            ai_df = ai_df[ai_df.state < state_cutoff]
        ai_df = ai_df.reset_index()
        ai_df_rs[f'r{r}'] = ai_df

    # print("ab initio dataframe", ai_df_rs)

    matches = ['t_1', 'doccp', 'sisj']  # onebody_params + twobody_params
    weights = [1 - w, w]

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
        beta,
        p,
        guess_params,
        lamb=lamb,
        niter_opt=niter_opt,
        tol_opt=tol_opt,
        maxfev_opt=maxfev_opt,
    )


def runInference(named_terms: str,
                 ai_dir: str,
                 model_descriptors: str,
                 inference_name: str,
                 nroots: int,
                 onebody_params: list[str],
                 twobody_params: list[str],
                 rs: list[float],
                 w: float = 0.1,
                 beta: float = 0.0,
                 minimum_1s_occupation: float = 0.91,
                 state_cutoff: int = None,
                 lamb: float = 1.0,
                 tmpdirname=None
                 ):
    """ Evaluates an inference of a given model with w=0.1.

    Parameters
    ----------
    named_terms : str
        File that holds the onebody and twobody terms.
    ai_dir : str
        File that holds the ab initio data. The descriptor information of the ab initio data. The keys should match model descriptors
    model_descriptors : str
        Name of file to store model in hdf5 format.
    inference_name : str
        Name of group inside hdf5 file to store inference evaluation.
    nroots : int
        Number of roots to solve the effective model for.
    onebody_params :list[str]
        List of the key names for the onebody parameters.
    twobody_params : list[str]
        List of the key names for the twobody parameters.
    train_rs : list[float]
        List of the atomic spacing, r, to optimize over.
    param_functions : str
        The function type of the parameter vs r function.
    w : float, optional
        The ratio of descriptor to spectrum loss value, by default 0.1.
    beta : float, optional
        Boltmann beta for boltmann weighting term, by default 0.0.
    minimum_1s_occupation : float, optional
        The minimum occupancy per site in the target space, by default 0.91
    state_cutoff : int, optional
        The state number of ab initio data that is the cutoff value, by default None.
    lamb : float, optional
        Strength of the intruder state penalty term, by default 1.0.

    Returns
    -------
    float
        Loss function dictionary.
    """

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
        ai_df = ai_df[ai_df.delta == 0.0]
        ai_df = ai_df[ai_df.trace >= len(onebody['trace'])*minimum_1s_occupation]
        if state_cutoff is not None:
            ai_df = ai_df[ai_df.state < state_cutoff]
        ai_df = ai_df.reset_index()
        ai_df_rs[f'r{r}'] = ai_df

    matches = ['t_1', 'doccp', 'sisj']  # onebody_params + twobody_params
    weights = [1 - w, w]

    params_dict = {}

    with h5py.File(model_descriptors, 'r') as f:
        for r in rs:
            param_list = []
            for parameter in onebody_params + twobody_params:
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
        weights,
        beta,
        params_dict,
        lamb=lamb,
    )


def make_name(parameters):
    return "_".join(parameters[0]) + "_" + "_".join(parameters[1])


def main(parameters, state_cutoff, w, beta, train_rs, niter_opt, tol_opt, maxfev_opt, nCV_iter, param_functions, lamb, df):
    with mlflow.start_run():
        # Write model and plots to temp dir
        with tempfile.TemporaryDirectory() as output_dir:
            model_files = []
            for i in range(nCV_iter):
                pname = make_name(parameters)
                df_tmp = df.copy()
                df_tmp = df_tmp[df_tmp['model_name'] == pname]
                df_tmp = df_tmp[np.isin(df_tmp['r'], train_rs)]
                df_tmp = df_tmp[df_tmp['beta'] == beta]
                if df_tmp.empty:
                    df_tmp = None
                else:
                    df_tmp = df_tmp.sort_values(by='r', ascending=True)
                #print(df) # to-do: put in warning statement if df has multiple entries for same r and beta and model_name
                dirname = os.path.join(output_dir, f"func_model_data_{state_cutoff}_{w}")
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                model_file_path = f"{dirname}/{pname}_{i}.hdf5"

                """ runCV(named_terms="hchain4_named_operators.hdf5",
                      ai_dir="ai_data/ai_descriptors_natoms4_nMO4.csv",
                      model_descriptors=model_file_path,
                      nroots=36,
                      onebody_params=parameters[0],
                      twobody_params=parameters[1],
                      train_rs=train_rs,
                      param_functions=param_functions,
                      w=w,
                      beta=beta,
                      p=0, # Set to 0, no CV for now
                      state_cutoff=state_cutoff,  #state_cutoff,
                      lamb=lamb,
                      niter_opt=niter_opt,
                      tol_opt=tol_opt,
                      maxfev_opt=maxfev_opt
                      )

                runInference(named_terms="hchain4_named_operators.hdf5",
                             ai_dir="ai_data/ai_descriptors_natoms4_nMO4.csv",
                             model_descriptors=model_file_path,
                             inference_name="natoms4_casci",
                             nroots=36,
                             onebody_params=parameters[0],
                             twobody_params=parameters[1],
                             rs=train_rs,
                             beta=beta,
                             state_cutoff=state_cutoff,
                             lamb=lamb,
                             ) """

                nMOs = 18
                basis = 'vtz'
                runCV(named_terms="hchain6_named_operators.hdf5",
                      ai_dir=f"ai_data/ai_descriptors_natoms6_nMO{nMOs}_basis{basis}.csv",
                      model_descriptors=model_file_path,
                      nroots=150,
                      onebody_params=parameters[0],
                      twobody_params=parameters[1],
                      train_rs=train_rs,
                      param_functions=param_functions,
                      w=w,
                      beta=beta,
                      p=1, # Set to 0, no CV for now
                      state_cutoff=state_cutoff,
                      lamb=lamb,
                      guess_params=df_tmp,
                      niter_opt=niter_opt,
                      tol_opt=tol_opt,
                      maxfev_opt=maxfev_opt
                      )

                runInference(named_terms="hchain6_named_operators.hdf5",
                             ai_dir=f"ai_data/ai_descriptors_natoms6_nMO{nMOs}_basis{basis}.csv",
                             model_descriptors=model_file_path,
                             inference_name=f"natoms6_casci_nMO{nMOs}_basis{basis}",
                             nroots=400,
                             onebody_params=parameters[0],
                             twobody_params=parameters[1],
                             rs=train_rs,
                             beta=beta,
                             state_cutoff=state_cutoff,
                             lamb=lamb,
                             )

                """ runInference(named_terms="hchain8_named_operators.hdf5",
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
                             )  """

                mlflow.log_artifact(model_file_path)
                model_files.append(model_file_path)
            plot_model(output_dir, model_files, [f"natoms6_casci_nMO{nMOs}_basis{basis}"], parameters)
            # plot_thermo(output_dir, model_files, [f"natoms6_casci_nMO{nMOs}_basis{basis}"], [f"ai_data/ai_descriptors_natoms6_nMO{nMOs}_basis{basis}.csv"], parameters)


if __name__ == "__main__":
    #temp = 10000000000000.0 #1000 # eV or T*kb
    #beta = 0.1
    #lamb = 1.0
    #main((['trace', 't_1'], ['doccp']), 150, 0.4, beta, [5.0], 100, 1e-9, 1000, 1, ['independent', 'independent', 'independent'], lamb)

    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=str, nargs="+")
    parser.add_argument("--train_rs", type=str)
    parser.add_argument("--state_cutoff")
    parser.add_argument("--w", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--lamb", type=float)
    parser.add_argument("--niter_opt", type=int)
    parser.add_argument("--tol_opt", type=float)
    parser.add_argument("--nCV_iter", type=int, default=1)
    parser.add_argument("--maxfev_opt", type=float, default=1000)
    parser.add_argument("--parameter_functions", type=str, nargs="+")

    args = parser.parse_args()

    parameters = (args.parameters[0].split(','), args.parameters[1].split(','))
    if args.state_cutoff == "None":
        state_cutoff = None
    else:
        state_cutoff = int(args.state_cutoff)

    w = args.w
    beta = args.beta
    lamb = args.lamb
    train_rs = [float(r) for r in args.train_rs.split(",")]
    niter_opt = args.niter_opt
    tol_opt = args.tol_opt
    nCV_iter = args.nCV_iter
    maxfev_opt = args.maxfev_opt
    param_functions = args.parameter_functions[0].split(',') + args.parameter_functions[1].split(',')

    df = pd.read_csv("model_parameters.csv")

    main(parameters, state_cutoff, w, beta, train_rs, niter_opt, tol_opt, maxfev_opt, nCV_iter, param_functions, lamb, df)
