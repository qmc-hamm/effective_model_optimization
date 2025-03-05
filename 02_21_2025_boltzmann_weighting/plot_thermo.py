import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py
import os
import numpy as np
from scipy.constants import k, eV

import mlflow

kb = k/eV

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

            train_rmse.append(f["Mean over r Spectrum RMSE - Train"][()])
            val_rmse.append(f["Mean over r Spectrum RMSE - Validation"][()])
            loss.append(f["loss"][()])

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

    mlflow.log_metrics({"CV-avg_mean-over-r_spectrum-rmse_ha_train": np.mean(train_rmse),
                        "CV-avg_mean-over-r_spectrum-rmse_ha_val": np.mean(val_rmse),
                        "CV-avg_loss": np.mean(loss),
                        })

    return pd.DataFrame(data)


def gather_model_energies(dirname, fnames, inference_names):

    data = []

    for i, fname in enumerate(fnames):
        with h5py.File(os.path.join(dirname, fname), 'r') as f:

            train_rs = f['train_rs'][()]

            all_rs = list(train_rs)
            all_rs.sort()

            r_inference = {}
            for inference_name in inference_names:
                r_inference[f"{inference_name}"] = f[f"inference_{inference_name}/rs"][()]

            for r in all_rs:
                for inference_name in inference_names:
                    if r in r_inference[f"{inference_name}"]:
                        f_energies = f[f"inference_{inference_name}/r{r}/test/descriptors/energy"][()]

                        for j, energy in enumerate(f_energies):
                            energies = {}
                            energies[f"r"] = r
                            energies[f"state"] = j
                            energies[f"energy (eV)"] = energy
                            energies[f"model number"] = i
                            data.append(energies)

    return data


# temperature should be something like 10 eV -> 100 eV
def compute_boltzmann_weights(energies, temperature:float):
    """
    temperature in eV
    """
    beta = 1.0/(temperature)
    boltzmann = np.exp(-beta*energies)
    Z = np.sum(boltzmann)
    return boltzmann/Z


def entropy(boltzmann_weights):
    return -np.sum(boltzmann_weights*np.log(boltzmann_weights))


def energy(boltzmann_weights, energies):
    return np.sum(boltzmann_weights*energies)


def free_energy(boltzmann_weights, energies, t):
    return np.sum(boltzmann_weights*energies) - t*kb*-np.sum(boltzmann_weights*np.log(boltzmann_weights))


def thermo_from_df(ai_energies:np.array, m_energies:np.array):
    T = np.linspace(0.1, 100, 1000)

    S_ai = np.asarray([entropy(compute_boltzmann_weights(ai_energies, t)) for t in T])
    U_ai = np.asarray([energy(compute_boltzmann_weights(ai_energies, t), ai_energies) for t in T])
    F_ai = np.asarray([free_energy(compute_boltzmann_weights(ai_energies, t), ai_energies, t) for t in T])

    S_m = np.asarray([entropy(compute_boltzmann_weights(m_energies, t)) for t in T])
    U_m = np.asarray([energy(compute_boltzmann_weights(m_energies, t), m_energies) for t in T])
    F_m = np.asarray([free_energy(compute_boltzmann_weights(m_energies, t), m_energies, t) for t in T])

    plt.plot(T, S_ai, label='ab initio')
    plt.plot(T, S_m, label='model')
    RMSE_S = np.sqrt(np.mean((S_ai-S_m)**2))
    plt.title(f'S RMSE : {np.round(RMSE_S, 2)}')
    plt.xlabel("Temperature (eV)")
    plt.ylabel("Entropy")
    plt.legend()
    plt.savefig("plots/T_S.png")
    plt.xlim(0, 10)
    plt.savefig("plots/T_S_zoomed.png")
    plt.figure()
    mlflow.log_artifact("plots/T_S_zoomed.png")

    plt.plot(T, np.exp(S_ai), label='ab initio')
    plt.plot(T, np.exp(S_m), label='model')
    plt.xlabel("Temperature (eV)")
    plt.ylabel("Effective number of states")
    plt.legend()
    plt.savefig("plots/T_n.png")
    plt.xlim(0, 10)
    plt.savefig("plots/T_n_zoomed.png")
    mlflow.log_artifact("plots/T_n_zoomed.png")
    plt.figure()

    plt.plot(T, U_ai, label='ab initio')
    plt.plot(T, U_m, label='model')
    RMSE_U = np.sqrt(np.mean((U_ai-U_m)**2))
    plt.title(f'U RMSE : {np.round(RMSE_U, 2)}')
    plt.xlabel("Temperature (eV)")
    plt.ylabel("Internal Energy (eV)")
    plt.legend()
    plt.savefig("plots/T_U.png")
    plt.xlim(0, 10)
    plt.savefig("plots/T_U_zoomed.png")
    plt.figure()
    mlflow.log_artifact("plots/T_U_zoomed.png")

    dT = np.diff(T, n=1)
    T_avg = (T[1:] + T[:-1])/2
    dU_ai = np.diff(U_ai, n=1)
    dU_m = np.diff(U_m, n=1)
    plt.plot(T_avg, dU_ai/dT, label='ab initio')
    plt.plot(T_avg, dU_m/dT, label='model')
    plt.xlabel("Temperature (eV)")
    plt.ylabel("Cv Heat Capacity (eV/K)")
    plt.legend()
    plt.savefig("plots/T_Cv_from_U.png")
    plt.xlim(0, 10)
    plt.savefig("plots/T_Cv_from_U_zoomed.png")
    mlflow.log_artifact("plots/T_Cv_from_U_zoomed.png")
    plt.figure()

    dS_ai_units = np.diff(kb*S_ai, n=1)
    dS_m_units = np.diff(kb*S_m, n=1)
    plt.plot(T_avg, T_avg*(dS_ai_units/dT), label='ab initio')
    plt.plot(T_avg, T_avg*(dS_m_units/dT), label='model')
    plt.xlabel("Temperature (eV)")
    plt.ylabel("Cv Heat Capacity eV/K")
    plt.legend()
    plt.savefig("plots/T_Cv_from_S.png")
    plt.xlim(0, 10)
    plt.savefig("plots/T_Cv_from_S_zoomed.png")
    plt.figure()

    d_ln_T = np.diff(np.log(T), n=1)
    dS_ai = np.diff(S_ai, n=1)
    dS_m = np.diff(S_m, n=1)
    plt.plot(T_avg, dS_ai/d_ln_T, label='ab initio')
    plt.plot(T_avg, dS_m/d_ln_T, label='model')
    plt.xlabel("Temperature (eV)")
    plt.ylabel("C* Heat Capacity")
    plt.legend()
    plt.savefig("plots/T_C_dimensionless.png")
    plt.xlim(0, 10)
    plt.savefig("plots/T_C_dimensionless_zoomed.png")
    plt.figure()

    plt.plot(T, F_ai, label='ab initio')
    plt.plot(T, F_m, label='model')
    plt.xlabel("Temperature (eV)")
    plt.ylabel("Free Energy (eV)")
    plt.legend()
    plt.savefig("plots/T_F.png")


    H_Cv = 14.3 #J/gK
    N = 4
    H_m = N*1.6735575e-24
    H_Cv_eV_K = (H_Cv*H_m)/eV

    #print("Heat capacity of hyrdogen in eV/K", H_Cv_eV_K)

    return


def gather_thermo_data(ai_energies:np.array, model_energies:pd.DataFrame):

    #print(model_energies)

    T = np.linspace(0.1, 100, 1000)

    S_ai = np.asarray([entropy(compute_boltzmann_weights(ai_energies, t)) for t in T])
    U_ai = np.asarray([energy(compute_boltzmann_weights(ai_energies, t), ai_energies) for t in T])
    F_ai = np.asarray([free_energy(compute_boltzmann_weights(ai_energies, t), ai_energies, t) for t in T])

    dT = np.diff(T, n=1)
    T_avg = (T[1:] + T[:-1])/2
    dU_ai = np.diff(U_ai, n=1)
    C_ai = dU_ai/dT

    T_avg = np.append(T_avg, [None])
    C_ai = np.append(C_ai, [None])

    thermo_data = []

    #print("saving dict data")
    for i in range(len(S_ai)):
        thermo = {}
        thermo['T (eV)'] = T[i]
        thermo['method'] = 'ab initio'
        thermo['model number'] = None
        thermo['S'] = S_ai[i] 
        thermo['U'] = U_ai[i] 
        thermo['F'] = F_ai[i]
        thermo['C'] = C_ai[i]
        thermo['T for C (eV)'] = T_avg[i]
        thermo_data.append(thermo)
    #print("end of saving dict data")

    model_numbers = np.unique(model_energies['model number'])

    for model_number in model_numbers:
        energies = model_energies[model_energies['model number'] == model_number]

        S_m = np.asarray([entropy(compute_boltzmann_weights(energies['energy (eV)'], t)) for t in T])
        U_m = np.asarray([energy(compute_boltzmann_weights(energies['energy (eV)'], t), energies['energy (eV)']) for t in T])
        F_m = np.asarray([free_energy(compute_boltzmann_weights(energies['energy (eV)'], t), energies['energy (eV)'], t) for t in T])
        
        dU_m = np.diff(U_m, n=1)
        C_m = dU_m/dT

        C_m = np.append(C_m, [None])

        for i in range(len(S_m)):
            thermo = {}
            thermo['T (eV)'] = T[i]
            thermo['method'] = 'model'
            thermo['model number'] = model_number
            thermo['S'] = S_m[i] 
            thermo['U'] = U_m[i] 
            thermo['F'] = F_m[i] 
            thermo['C'] = C_m[i]
            thermo['T for C (eV)'] = T_avg[i]
            thermo_data.append(thermo)

    return pd.DataFrame(thermo_data)


def plot_thermo(dirname: str, fnames: list[str], inference_names: list[str], abinitio_names: list[str], parameters: tuple[list[str], list[str]]) -> str:
    #df = gather_data_to_plot(dirname, fnames, inference_names, parameters[0] + parameters[1])
    r = 3.0
    Tfit = 10 #eV

    #print("file names", fnames)

    if not os.path.exists("plots"):
        os.makedirs("plots")

    for i, name in enumerate(inference_names):
        df = pd.read_csv(abinitio_names[i])
        df = df[df.r == r].reset_index()

        model_energies = gather_model_energies(".", fnames, inference_names)

        df_model = pd.DataFrame(model_energies)
        df_model = df_model[df_model.r == r].reset_index()

        thermo_df = gather_thermo_data(df['energy'], df_model)

        thermo_df.to_csv("plots/thermo_df.csv")
        mlflow.log_artifact("plots/thermo_df.csv")

        sns.lineplot(data=thermo_df[thermo_df["method"]=="ab initio"], x='T (eV)', y='S', hue='method')
        sns.lineplot(data=thermo_df[thermo_df["method"]=="model"], x='T (eV)', y='S', hue='model number', palette="dark:salmon")
        plt.xlabel("Temperature (eV)")
        plt.ylabel("Entropy")
        plt.xlim(0, 10)
        plt.savefig("plots/T_S_zoomed.png", dpi=300)
        mlflow.log_artifact("plots/T_S_zoomed.png")
        plt.figure()

        #sns.lineplot(data=thermo_df, x='T (eV)', y='U', hue='method')
        sns.lineplot(data=thermo_df[thermo_df["method"]=="ab initio"], x='T (eV)', y='U', hue='method')
        sns.lineplot(data=thermo_df[thermo_df["method"]=="model"], x='T (eV)', y='U', hue='model number', palette="dark:salmon")
        plt.xlabel("Temperature (eV)")
        plt.ylabel("Potential Energy (eV)")
        plt.xlim(0, 10)
        plt.savefig("plots/T_U_zoomed.png", dpi=300)
        mlflow.log_artifact("plots/T_U_zoomed.png")
        plt.figure()

        #sns.lineplot(data=thermo_df, x='T for C (eV)', y='C', hue='method')
        sns.lineplot(data=thermo_df[thermo_df["method"]=="ab initio"], x='T for C (eV)', y='C', hue='method')
        sns.lineplot(data=thermo_df[thermo_df["method"]=="model"], x='T for C (eV)', y='C', hue='model number', palette="dark:salmon")
        plt.xlabel("Temperature (eV)")
        plt.ylabel("Heat Capacity")
        plt.xlim(0, 10)
        plt.savefig("plots/T_C_zoomed.png", dpi=300)
        mlflow.log_artifact("plots/T_C_zoomed.png")
        plt.figure()

        #thermo_from_df(df['energy'], model_energies[f"r{r}"])

    return


if __name__ == "__main__":

    state_cutoff = 10
    w0 = 0.9
    parameters = (['E0', 't'], ['U'])
    pname = make_name(parameters)
    i = 0

    #files = [f"func_model_data_{state_cutoff}_{w0}/{pname}_{i}.hdf5"]
    #REPLACE #files = ["mlruns/0/c927c6e17bec44adbe95f0de3b97e8d6/artifacts/trace_t_1_doccp_0.hdf5"]
    
    CV_iter = 10

    files = []
    for i in range(CV_iter):
        file = f'mlruns/0/bb25e14cfb52481d927ec096ffb1e33d/artifacts/trace_t_1_doccp_{i}.hdf5'
        files.append(file)

    inference_names = ["natoms4_casscf"]#, "natoms8_casscf", "natoms8_vmc"]

    abinitio_names = ["ai_data/hchain4.csv"]

    plot_thermo(".", files, inference_names, abinitio_names, (['E0', 't'], ['U']))