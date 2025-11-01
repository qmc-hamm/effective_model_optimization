import argparse
import itertools
import os
from typing import Optional, Dict, Union, List

import mlflow

# Hyperparameters
parameter_sets = [
    # Three terms
    (['trace', 't_1'], ['doccp']),
    #(['trace'], ['sisj', 'doccp']),
    #(['trace'], ['doccp', 'hophop']),
    #(['trace'], ['doccp', 'denhop_three_NN']),
    #(['trace'], ['doccp', 'denhop_two_NN']),
    #(['trace'], ['doccp', 'hophop_three_NN']),

    # Four Terms
    (['trace', 't_1'], ['doccp', 'sisj']),
    (['trace', 't_1'], ['doccp', 'densityNN']),
    (['trace', 't_1', 't_2'], ['doccp']),
    #(['trace', 't_1'], ['doccp', 'hophop']),
    #(['trace', 't_1'], ['doccp', 'hophop_four_NN_1']),
    #(['trace', 't_1'], ['doccp', 'hophop_four_NN_2']),
    
    # Five Terms
    (['trace', 't_1'], ['doccp', 'sisj', 'densityNN']),
    #(['trace', 't_1', 't_2'], ['doccp', 'densityNN']),
    #(['trace', 't_1'], ['doccp', 'densityNN', 'denhop_three_NN']),
    #(['trace', 't_1'], ['doccp', 'densityNN', 'hophop_four_NN_1']),
    #(['trace', 't_1'], ['doccp', 'densityNN', 'hophop_four_NN_2']),
    #(['trace', 't_1'], ['doccp', 'densityNN', 'hophop']),
]
param_function_sets = [
    {'trace':'independent', 'e_end':'independent', 'e_center':'independent', 'doccp':'independent', 
     't_1':'independent', 't_2':'independent', 't_3':'independent', 'v':'independent', 'sisj':'independent',
     'densityNN':'independent', 'exchange':'independent', 'hophop':'independent', 'denhop_three_NN':'independent',
     'denhop_two_NN':'independent', 'hophop_four_NN_1':'independent', 'hophop_four_NN_2':'independent', 
     'hophop_three_NN':'independent'},

#    {'trace':'func_E0', 'doccp':'independent', 't_1':'independent'},
#    {'trace':'func_E0', 'doccp':'independent', 't_1':'polynomial5'},
#    {'trace': 'func_E0', 'doccp': 'independent', 't_1': 'exponential'},
]
rs_set = [
    #[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4, 5.0]
    [3.2, 4.0, 4.8] # 3.0
    #[2.2, 2.8, 3.2, 3.6, 4.0, 4.4]  # Test Workflow
]
state_cutoffs = [
    #None # Test Workflow
    #14 #8, 10, 12, 14
    50
]
ws = [
    #1.0, 0.9, 0.8, 0.7, 0.6
#    1.0, 0.95, 0.9, 0.85, 0.8
#    1.0, 0.9, 0.8, 0.7, 0.6
    0.05 #, 0.1, 0.2, 0.3, 0.4  # Test Workflow 0.4
]

betas = [
        0.05, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0#, 0.5, 1.0, 2.0
         #0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
         ]

lambdas =[
    1.0
]


def prepare_mlflow_params(
        state_cutoff: Optional[float] = None,
        train_rs: Optional[List[float]] = None,
        w: Optional[float] = None,
        beta: Optional[float] = None,
        lamb: Optional[float] = None,
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
        w: Weight value of physical descriptors in loss function
        beta: Sets Boltzmann Weights
        lamb: Weight in front of penalty term, default is 1.0
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
    if w is not None:
        params["w"] = str(w)
    if beta is not None:
        params["beta"] = str(beta)
    if lamb is not None:
        params["lamb"] = str(lamb)
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
    for parameters, parameter_function_dict, train_rs, state_cutoff, w, beta, lamb in itertools.product(parameter_sets,
                                                                                             param_function_sets,
                                                                                             rs_set,
                                                                                             state_cutoffs,
                                                                                             ws,
                                                                                             betas,
                                                                                             lambdas):
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
            w=w,
            beta=beta,
            lamb=lamb,
        )

        jobs.append(run_train(
            experiment_id,
            training_backend=args.training_backend,
            training_backend_config=args.training_backend_config,
            run_params=job_params)
        )
    results = map(lambda job: job.wait(), jobs)
    print(list(results))
