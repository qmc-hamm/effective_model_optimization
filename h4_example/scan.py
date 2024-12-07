import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Param, RunTag


def run_train(experiment_id,
              iters,
              training_backend,
              training_backend_config,
              parent_run_id=None):
    print(f"Starting run with backend {training_backend} and config {training_backend_config}")

    kwargs = {
        "uri": os.path.dirname(os.path.realpath(__file__)),
        "entry_point": "main",
        "parameters": {
            "niter_opt": str(iters),
            "tol_opt": str(iters),
            "nCV_iter": str(iters)
        },
        "experiment_id": experiment_id,
        "synchronous": True if training_backend == "local" else False,
        "backend": training_backend,
    }

    if training_backend != "local":
        kwargs["backend_config"] = training_backend_config

    p = mlflow.projects.run(**kwargs)

    MlflowClient().log_batch(run_id=p.run_id, metrics=[],
                             params=[Param("niter_opt", str(iters)),
                                     Param("tol_opt", str(iters)),
                                     Param("nCV_iter", str(iters))]),
                             # tags=[RunTag(mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID, parent_run_id)])

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

    for iters in range(1,6):
        jobs.append(run_train(
            experiment_id,
            iters=iters*100,
            training_backend=args.training_backend,
            training_backend_config=args.training_backend_config,
            parent_run_id=provided_run_id)
        )
        results = map(lambda job: job.wait(), jobs)
        print(list(results))
