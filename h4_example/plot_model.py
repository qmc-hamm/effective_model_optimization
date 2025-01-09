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


def gather_data_to_plot(dirname, fnames, parameters):
    data = []

    train_rmse = []
    val_rmse = []
    test_rmse = []
    loss = []

    for i, fname in enumerate(fnames):
        with h5py.File(os.path.join(dirname, fname), 'r') as f:
            train_rs = f['train_rs'][()]
            test_rs = f['test_rs'][()]

            all_rs = list(train_rs)+list(test_rs)
            all_rs.sort()

            train_rmse.append( f["Mean over r Spectrum RMSE (Ha) - Train"][()] )
            val_rmse.append( f["Mean over r Spectrum RMSE (Ha) - Validation"][()] )
            test_rmse.append( f["Mean over r Spectrum RMSE (Ha) - Test"][()] )
            loss.append( f["loss"][()] )

            for r in all_rs:
                data_r = {}
                data_r['r (Bohr)'] = r

                data_r['CV iteration'] = i

                data_r['loss'] = f['loss'][()]

                if r in train_rs:
                    data_r['Spectrum RMSE Train (Ha)'] = f[f'r{r}/Spectrum RMSE Train (Ha)'][()]
                    data_r['Spectrum RMSE Val (Ha)'] = f[f'r{r}/Spectrum RMSE Val (Ha)'][()]
                    data_r['Spectrum RMSE Test (Ha)'] = None
                elif r in test_rs:
                    data_r['Spectrum RMSE Train (Ha)'] = None
                    data_r['Spectrum RMSE Val (Ha)'] = None
                    data_r['Spectrum RMSE Test (Ha)'] = f[f'r{r}/Spectrum RMSE (Ha)'][()]

                for parameter in parameters:
                    data_r[f'RDMD {parameter} (Ha)'] = f[f'r{r}/rdmd_params/{parameter}'][()]
                    data_r[f'DMD {parameter} (Ha)'] = f[f'r{r}/dmd_params/{parameter}'][()]

                data.append(data_r)

    mlflow.log_metrics({
    "CV-avg_mean-over-r_spectrum-rmse_ha_train": np.mean(train_rmse),
    "CV-avg_mean-over-r_spectrum-rmse_ha_val": np.mean(val_rmse),
    "CV-avg_mean-over-r_spectrum-rmse_ha_test": np.mean(test_rmse),
    "CV-avg_loss": np.mean(loss),
    })

    return pd.DataFrame(data)


def plot_model(dirname: str, fnames: list[str], parameters: tuple[list[str], list[str]]) -> str:
    df = gather_data_to_plot(dirname, fnames, parameters[0]+parameters[1])
    
    csv_file = f'{dirname}/Processed_CV_data.csv'
    df.to_csv(csv_file)
    mlflow.log_artifact(csv_file)

    plot_file = f'{dirname}/Spectrum_RMSEvs_r.png'
    sns.lineplot(data=df, x='r (Bohr)', y='Spectrum RMSE Train (Ha)', label="Train")
    sns.lineplot(data=df, x='r (Bohr)', y='Spectrum RMSE Val (Ha)', label="Validation")
    sns.lineplot(data=df, x='r (Bohr)', y='Spectrum RMSE Test (Ha)', label="Test")
    plt.ylabel("Spectrum RMSE (Ha)")
    plt.legend()
    plt.savefig(plot_file, dpi=200)
    plt.clf()

    mlflow.log_artifact(plot_file)

    for parameter in parameters[0]+parameters[1]:
        plot_file2 = f'{dirname}/RDMD_{parameter}vs_r.png'
        sns.lineplot(data=df, x='r (Bohr)', y=f'RDMD {parameter} (Ha)', label="RDMD")
        sns.lineplot(data=df, x='r (Bohr)', y=f'DMD {parameter} (Ha)', label="DMD")
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

    plot_model(".", files,
               (['E0', 't'], ['U']) )

    
        