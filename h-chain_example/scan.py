import argparse
import itertools
import os
from typing import Optional, Dict, Union, List

import mlflow

# Hyperparameters
parameter_sets = [
    #(['trace'], ['sisj', 'doccp']),
    (['trace', 't_1'], ['doccp']),
    #(['E0', 't'], ['U']),
    # (['E0', 't', 'tdiag'], ['U']),
    #(['E0', 't'], ['U', 'V']),
    #(['E0', 't'], ['U', 'J']),
    #(['E0', 't'], ['U', 'V', 'J']),
    # (['E0', 't'], ['J']),
    # (['E0'], ['U', 'J']),
    # (['E0', 't','tdiag'], ['U', 'J']),
    # (['E0', 't','tdiag'], ['U', 'V']),
    # (['E0', 't','tdiag'], ['U', 'V','J']),
]
param_function_sets = [
#    {'trace':'independent', 'doccp':'independent', 't_1':'independent'},
#    {'trace':'func_E0', 'doccp':'independent', 't_1':'independent'},
#    {'trace':'func_E0', 'doccp':'independent', 't_1':'polynomial5'},
    {'trace':'func_E0', 'doccp':'independent', 't_1':'exponential'},
]
rs_set = [
    #[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4, 5.0]
    [2.8, 3.0, 3.2, 3.6, 4.0, 4.4, 5.0]
    #[2.2, 2.8, 3.2, 3.6, 4.0, 4.4]  # Test Workflow
]
state_cutoffs = [
    None # Test Workflow
    #8, 10, 12, 14
]
w0s = [
    1.0, 0.9, 0.8
#    1.0, 0.95, 0.9, 0.85, 0.8
#    1.0, 0.9, 0.8, 0.7, 0.6
#    0.8  # Test Workflow
]


def prepare_mlflow_params(
        state_cutoff: Optional[float] = None,
        train_rs: Optional[List[float]] = None,
        w0: Optional[float] = None,
        parameter0: Optional[List[str]] = None,
        parameter1: Optional[List[str]] = None,
        niter_opt: Optional[float] = None,
        tol_opt: Optional[float] = None,
        maxfev_opt: Optional[float] = None,
        nCV_iter: Optional[float] = None,
        parameter_function0: Optional[List[str]] = None,
        parameter_function1: Optional[List[str]] = None,
) -> Dict[str, Union[str, bool]]:
    """
    Prepare parameters for Cross Validation training run.

    Args:
        state_cutoff: Cutoff state value
        train_rs: Comma-separated string of values
        w0: Initial weight value
        parameter0: First parameter string
        parameter1: Second parameter string
        niter_opt: Number of optimization iterations
        tol_opt: Optimization tolerance
        maxfev_opt: Maximum function evaluations
        nCV_iter: Number of cross-validation iterations
        parameter_function0: First parameter function name string
        parameter_function1: Second parameter function name string

    Returns:
        Dictionary with MLflow run configuration parameters. The dict will
        only have entries for the non-None parameters.
    """
    # Prepare parameters dictionary, converting non-None values to strings
    params = {}

    if state_cutoff is not None:
        params["state_cutoff"] = str(state_cutoff)
    if train_rs is not None:
        params["train_rs"] = ",".join([str(r) for r in train_rs])
    if w0 is not None:
        params["w0"] = str(w0)
    if parameter0 is not None:
        params["parameter0"] = ",".join(parameter0)
    if parameter1 is not None:
        params["parameter1"] = ",".join(parameter1)
    if parameter_function0 is not None:
        params["parameter_function0"] = ",".join(parameter_function0)
    if parameter_function1 is not None:
        params["parameter_function1"] = ",".join(parameter_function1)
    if niter_opt is not None:
        params["niter_opt"] = str(niter_opt)
    if tol_opt is not None:
        params["tol_opt"] = str(tol_opt)
    if maxfev_opt is not None:
        params["maxfev_opt"] = str(maxfev_opt)
    if nCV_iter is not None:
        params["nCV_iter"] = str(nCV_iter)

    return params


def run_train(experiment_id,
              training_backend,
              training_backend_config,
              run_params: dict):
    """
    Run Cross Validation training job.
    Parameters
    ----------
    experiment_id
    training_backend: The name of the MLFlow backend. Use local or slurm
    training_backend_config: Json file to pass to the backend. Leave empty for local backend
    run_params: Dictionary with MLflow run configuration parameters

    Returns
    -------
        :py:class:`mlflow.projects.SubmittedRun` exposing information (e.g. run ID)
        about the launched run.

    """

    # Prepare full kwargs dictionary for MLflow run
    kwargs = {
        "experiment_id": experiment_id,
        "uri": os.path.dirname(os.path.realpath(__file__)),
        "entry_point": "main",
        "parameters": run_params,
        "synchronous": True if training_backend == "local" else False,
        "backend": training_backend
    }

    if training_backend != "local":
        kwargs["backend_config"] = training_backend_config

    p = mlflow.projects.run(**kwargs)
    return p

parser = argparse.ArgumentParser()
parser.add_argument("--training_backend", type=str)
parser.add_argument("--training_backend_config",
                    type=str,
                    required=False,
                    default=None)

args = parser.parse_args()
provided_run_id = os.environ.get("MLFLOW_RUN_ID", None)
with mlflow.start_run(run_id=provided_run_id) as run:
    print("Search is run_id ", run.info.run_id)
    experiment_id = run.info.experiment_id
    jobs = []

    # Hyperparameter sweep step
    for parameters, parameter_function_dict, train_rs, state_cutoff, w0 in itertools.product(parameter_sets,
                                                              param_function_sets,
                                                              rs_set,
                                                              state_cutoffs,
                                                              w0s):
        
        param_functions = [[],[]]
        for param in parameters[0]:
            param_functions[0].append(parameter_function_dict[param])
        for param in parameters[1]:
            param_functions[1].append(parameter_function_dict[param])

        job_params = prepare_mlflow_params(
            parameter0=parameters[0],
            parameter1=parameters[1],
            parameter_function0=param_functions[0],
            parameter_function1=param_functions[1],
            train_rs=train_rs,
            state_cutoff=state_cutoff,
            w0=w0
        )

        jobs.append(run_train(
            experiment_id,
            training_backend=args.training_backend,
            training_backend_config=args.training_backend_config,
            run_params=job_params)
        )
    results = map(lambda job: job.wait(), jobs)
    print(list(results))
