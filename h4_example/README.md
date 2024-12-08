# Effective Model Optimization Using MLFlow

This project leverages MLflow to create effective model Hamiltonians 
with uncertainty analysis. The project can be executed using either a 
Python virtual environment or a Conda environment, depending on your system setup.

## Project Structure

- **MLProject File**: Defines the project name, environments, and entry points.
- **Python Environment**: Specified in `python_env.yaml`.
- **Conda Environment**: Optionally specified in `conda.yaml` for HPC setups.

## Entry Points

### Main

The main entry point runs a single cross-validation training step.

- **Parameters**:
  - `state_cutoff`: (float) Default is 10.
  - `rs`: (str) Default is "2.2, 2.8, 3.2, 3.6, 4.0, 4.4".
  - `w0`: (float) Default is 0.9.
  - `parameter0`: (string) Default is "E0,t".
  - `parameter1`: (string) Default is "U".
  - Debugging parameters:
    - `niter_opt`: (float) Default is 1.
    - `tol_opt`: (float) Default is 1.
    - `maxfev_opt`: (float) Default is 1.
    - `nCV_iter`: (float) Default is 1.

- **Command**:
  ```bash
  python cross_validation_function.py \
      --state_cutoff {state_cutoff} \
      --rs {rs} \
      --w0 {w0} \
      --niter_opt {niter_opt} \
      --tol_opt {tol_opt} \
      --maxfev_opt {maxfev_opt} \
      --parameters {parameter0} {parameter1}
  ```

### Scan

The scan entry point runs a parameter space scan, launching multiple 
cross-validation training steps in parallel.

- **Parameters**:
  - `training_backend`: (str) Default is "local". Options are "local" or "slurm".
  - `training_backend_config`: (str) Configuration file for slurm jobs if applicable.

- **Command**:
  ```bash
  python scan.py \
      --training_backend {training_backend} \
      --training_backend_config {training_backend_config}
  ```

## Getting Started

### Prerequisites

The MLFlow library requires a currently-supported version of python. We only need the
skinny version of this library for our training apps.
Installation can be done via pip:

```bash
pip install mlflow-skinny
```

### Running the Project

1. **Set Up Environment**: Choose between Python or Conda environment as specified in the MLProject file.
   - For Python: Use the `python_env.yaml` file to set up your environment.
   - For Conda: Use the `conda.yaml` file if running on HPC.
2. **Set Tracking Server Environmment**: Export the MLFLOW_TRACKING_URI environment variable to specify the location of the MLflow server. You can use the public MLflow server:
```bash
export MLFLOW_TRACKING_URI=https://<<MLFlow Tracking Server>>
```
3. **Execute Main Entry Point**: This is for specifying particylar parameters for a single model run.
   ```bash
   mlflow run . -P state_cutoff=10 -P rs="2.2,2.8" -P w0=0.9
   ```
4. **Execute Scan Entry Point**: First edit the `scan.py` file to specify the parameter space to scan. Then run the scan entry point.
   ```bash
   mlflow run . -e scan -P training_backend=local
   ```

## Running a Scan on Campus Cluster
You can use the slurm backend to run a scan on a campus cluster. This is run
from the CampusCluster login node. First, edit the `scan.py` file to specify the 
parameter space to scan. Then run the scan entry point with the slurm backend 
configuration file.
```bash
mlflow run . -e scan -P training_backend=slurm -P training_backend_config=cpu-config.yaml
```

## Notes

- Adjust parameters as needed for your specific use case.
- Ensure that all dependencies are correctly specified in the environment files for seamless execution.
- For distributed training using slurm, ensure that the configuration file is properly set up and referenced in the command.

This README provides an overview of how to use MLflow with this project for effective 
model optimization and uncertainty analysis. Adjust configurations as necessary to 
fit your computational environment and research needs.
