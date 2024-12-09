import tempfile

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py
import os


def make_name(parameters):
    return "_".join(parameters[0]) + "_" + "_".join(parameters[1])


def gather_data_to_plot(dirname, fname, parameters):
    data = []

    with h5py.File(os.path.join(dirname, fname), 'r') as f:
        rs = f['rs'][()]

        for r in rs:
            data_r = {}
            data_r['r (Bohr)'] = r

            data_r['Spectrum RMSE Train (Ha)'] = f[f'r{r}/Spectrum RMSE Train (Ha)'][()]
            data_r['Spectrum RMSE Val (Ha)'] = f[f'r{r}/Spectrum RMSE Val (Ha)'][()]

            for parameter in parameters:
                data_r[f'{parameter} (Ha)'] = f[f'r{r}/rdmd_params/{parameter}'][()]

            data.append(data_r)

    return pd.DataFrame(data)


def plot_model(dirname: str, fname: str, parameters: tuple[list[str], list[str]]) -> str:
    df = gather_data_to_plot(dirname, fname, parameters[0]+parameters[1])

    plot_file = f'{dirname}/Spectrum_RMSEvs_r.png'
    sns.lineplot(data=df, x='r (Bohr)', y='Spectrum RMSE Train (Ha)', label="Train")
    sns.lineplot(data=df, x='r (Bohr)', y='Spectrum RMSE Val (Ha)', label="Validation")
    plt.legend()
    plt.savefig(plot_file)
    plt.clf()

    for parameter in parameters[0]+parameters[1]:
        sns.lineplot(data=df, x='r (Bohr)', y=f'{parameter} (Ha)')
        plt.savefig(f'{dirname}/{parameter}vs_r.png')
        plt.clf()

    return plot_file


if __name__ == "__main__":

    state_cutoff = 10
    w0 = 0.9
    parameters = (['E0', 't'], ['U'])
    pname = make_name(parameters)
    i = 0

    plot_model(".", f"func_model_data_{state_cutoff}_{w0}/{pname}_{i}.hdf5",
               (['E0', 't'], ['U']) )

    
        