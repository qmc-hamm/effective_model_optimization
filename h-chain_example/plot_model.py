import tempfile

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py
import os
import numpy as np

import mlflow


def make_name(parameters):
    return "_".join(parameters[0]) + "_" + "_".join(parameters[1])


def gather_data_to_plot(dirname, fnames, inference_names, parameters):
    data = []

    train_rmse = []
    val_rmse = []
    loss = []

    for i, fname in enumerate(fnames):
        with h5py.File(os.path.join(dirname, fname), 'r') as f:
            train_rs = f['train_rs'][()]

            all_rs = list(train_rs)
            all_rs.sort()

            train_rmse.append( f["Mean over r Spectrum RMSE - Train"][()] )
            val_rmse.append( f["Mean over r Spectrum RMSE - Validation"][()] )
            loss.append( f["loss"][()] )

            r_inference = {}
            for inference_name in inference_names:
                r_inference[f"{inference_name}"] = f[f"inference_{inference_name}/rs"][()]

            for r in all_rs:
                data_r = {}
                data_r['r (Bohr)'] = r

                data_r['CV iteration'] = i

                data_r['loss'] = f['loss'][()]

                if r in train_rs:
                    data_r['Spectrum RMSE Train (eV)'] = f[f'r{r}/Spectrum RMSE Train'][()]
                    data_r['Spectrum RMSE Val (eV)'] = f[f'r{r}/Spectrum RMSE Val'][()]

                for parameter in parameters:
                    data_r[f'RDMD {parameter} (eV)'] = f[f'r{r}/rdmd_params/{parameter}'][()]
                    data_r[f'DMD {parameter} (eV)'] = f[f'r{r}/dmd_params/{parameter}'][()]

                for inference_name in inference_names:
                    if r in r_inference[f"{inference_name}"]:
                        data_r[f'Spectrum RMSE Test {inference_name} (eV)'] = f[f'inference_{inference_name}/r{r}/Spectrum RMSE'][()]
                
                data.append(data_r)

    mlflow.log_metrics({
    "CV-avg_mean-over-r_spectrum-rmse_ha_train": np.mean(train_rmse),
    "CV-avg_mean-over-r_spectrum-rmse_ha_val": np.mean(val_rmse),
    "CV-avg_loss": np.mean(loss),
    })

    return pd.DataFrame(data)


def plot_model(dirname: str, fnames: list[str], inference_names:list[str], parameters: tuple[list[str], list[str]]) -> str:
    df = gather_data_to_plot(dirname, fnames, inference_names, parameters[0]+parameters[1])
    
    csv_file = f'{dirname}/Processed_CV_data.csv'
    df.to_csv(csv_file)
    mlflow.log_artifact(csv_file)

    plot_file = f'{dirname}/Spectrum_RMSEvs_r.png'
    sns.lineplot(data=df, x='r (Bohr)', y='Spectrum RMSE Train (eV)', label="Train")
    sns.lineplot(data=df, x='r (Bohr)', y='Spectrum RMSE Val (eV)', label="Validation")
    for inference_name in inference_names:
        sns.lineplot(data=df, x='r (Bohr)', y=f'Spectrum RMSE Test {inference_name} (eV)', label=f"Test {inference_name}")
    plt.ylabel("Spectrum RMSE (eV)")
    plt.legend()
    plt.savefig(plot_file, dpi=200)
    plt.clf()

    mlflow.log_artifact(plot_file)

    for parameter in parameters[0]+parameters[1]:
        plot_file2 = f'{dirname}/RDMD_{parameter}vs_r.png'
        sns.lineplot(data=df, x='r (Bohr)', y=f'RDMD {parameter} (eV)', label="RDMD")
        sns.lineplot(data=df, x='r (Bohr)', y=f'DMD {parameter} (eV)', label="DMD")
        plt.savefig(plot_file2, dpi=200)
        plt.clf()

        mlflow.log_artifact(plot_file2)

    return


if __name__ == "__main__":

    state_cutoff = 10
    w0 = 0.9
    parameters = (['E0', 't'], ['U'])
    pname = make_name(parameters)
    i = 0

    files = [f"func_model_data_{state_cutoff}_{w0}/{pname}_{i}.hdf5"]

    inference_names = ["natoms6_casscf", "natoms8_casscf", "natoms8_vmc"]

    plot_model(".", files, inference_names,
               (['E0', 't'], ['U']) )

    
        