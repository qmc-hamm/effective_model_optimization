import os

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Param, RunTag


def run_train(experiment_id, iters, backend_config="slurm_config.json", parent_run_id=None):
    p = mlflow.projects.run(
        uri=os.path.dirname(os.path.realpath(__file__)),
        entry_point="main",
        parameters={
            "niter_opt": str(iters),
            "tol_opt": str(iters),
            "nCV_iter": str(iters)
        },
        experiment_id=experiment_id,
        synchronous=True,
        backend="local",
        # backend_config=backend_config
    )
    MlflowClient().log_batch(run_id=p.run_id, metrics=[],
                             params=[Param("niter_opt", str(iters)),
                                     Param("tol_opt", str(iters)),
                                     Param("nCV_iter", str(iters))]),
                             # tags=[RunTag(mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID, parent_run_id)])

    return p

provided_run_id = os.environ.get("MLFLOW_RUN_ID", None)
with mlflow.start_run(run_id=provided_run_id) as run:
    print("Search is run_id ", run.info.run_id)
    experiment_id = run.info.experiment_id
    jobs = []

    for iters in range(1,6):
        jobs.append(run_train(
            experiment_id,
            iters=iters*100,
            parent_run_id=provided_run_id)
        )
        results = map(lambda job: job.wait(), jobs)
        print(list(results))
