import pandas as pd
import statsmodels.api as sm
import h5py
import numpy as np
from pyscf import fci
from rich import print
from scipy.optimize import linear_sum_assignment, minimize, basinhopping
import scipy.linalg
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import time

eV2Ha = 0.0367492929

cache = {}  # For speedup during mulitiple ED during model optimization


# This section as all the information for solving with <psi_i|rho|psi_j>
def eff_model_trans_solver(h1, h2, norb=4, nelec=(2, 2), nroots=36):
    """Solves a Effective Model Hamiltonian with an FCI approach.
    Meant to be used in with hamiltonian_symmetry.py .
    """
    e, fcivec = fci.direct_spin1.kernel(
        h1, h2, norb, nelec, nroots=nroots, max_space=30, max_cycle=100, orbsym=None
    )

    n_rdm1s = np.zeros((nroots, nroots, 2, norb, norb))
    n_rdm2s = np.zeros((nroots, nroots, 4, norb, norb, norb, norb))
    for root1 in range(nroots):
        # rdm1s, rdm2s =  fci.direct_spin1.make_rdm12s(fcivec[root1], norb=norb, nelec=nelec)
        for root2 in range(nroots):
            rdm1s, rdm2s = fci.direct_spin1.trans_rdm12s(
                fcivec[root1], fcivec[root2], norb=norb, nelec=nelec
            )
            for i, r in enumerate(rdm1s):
                n_rdm1s[root1, root2, i, :, :] = r
            for i, r in enumerate(rdm2s):
                n_rdm2s[root1, root2, i, ...] = r
    return e, fcivec, n_rdm1s, n_rdm2s


def solve_effective_trans_hamiltonian(
    onebody: dict, twobody: dict, parameters: list, nroots: int
) -> pd.DataFrame:
    norb = 4  # should be passed in
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb, norb, norb, norb))
    for k in onebody.keys():
        if k in parameters.keys():
            h1 += parameters[k] * onebody[k]
    for k in twobody.keys():
        if k in parameters.keys():
            h2 += parameters[k] * twobody[k]
            # Changed
    h_eff_energies, h_eff_fcivec, rdm1, rdm2 = eff_model_trans_solver(
        h1, h2, norb=norb, nelec=(2, 2), nroots=nroots
    )

    print("lengths of fcivec ", len(h_eff_energies), len(h_eff_fcivec[0]))

    descriptors = {}
    descriptors["energy"] = np.diag(h_eff_energies)
    for k, it in onebody.items():
        print(it.shape, rdm1.shape)
        descriptors[k] = np.einsum("mn,ijsmn -> ij", it, rdm1)
    for k, it in twobody.items():
        descriptors[k] = np.einsum(
            "mnop,ijsmnop -> ij", it, rdm2[:, :, [0, 1, 3], :, :, :, :]
        )
    return descriptors


# This section is only <psi_i|pho|psi_i>, computationally faster
def eff_model_solver(h1, h2, norb=4, nelec=(2, 2), nroots=36, ci0: np.ndarray = None):
    """Solves a Effective Model Hamiltonian with an FCI approach.
    Meant to be used in with hamiltonian_symmetry.py .
    """
    e, fcivec = fci.direct_spin1.kernel(
        h1,
        h2,
        norb,
        nelec,
        nroots=nroots,
        max_space=30,
        max_cycle=100,
        orbsym=None,
        davidson_only=True,
        ci0=ci0,
    )  # These two can help speed up the solver

    n_rdm1s = np.zeros((nroots, 2, norb, norb))
    n_rdm2s = np.zeros((nroots, 4, norb, norb, norb, norb))
    for root1 in range(nroots):
        rdm1s, rdm2s = fci.direct_spin1.make_rdm12s(
            fcivec[root1], norb=norb, nelec=nelec
        )
        for i, r in enumerate(rdm1s):
            n_rdm1s[root1, i, :, :] = r
        for i, r in enumerate(rdm2s):
            n_rdm2s[root1, i, ...] = r
    return e, fcivec, n_rdm1s, n_rdm2s


def solve_effective_hamiltonian(
    onebody: dict, twobody: dict, parameters: list, nroots: int, ci0: np.ndarray = None
) -> pd.DataFrame:
    norb = 4
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb, norb, norb, norb))
    for k in onebody.keys():
        if k in parameters.keys():
            h1 += parameters[k] * onebody[k]
    for k in twobody.keys():
        if k in parameters.keys():
            h2 += parameters[k] * twobody[k]

    h_eff_energies, h_eff_fcivec, rdm1, rdm2 = eff_model_solver(
        h1, h2, norb=norb, nelec=(2, 2), nroots=nroots, ci0=ci0
    )

    # print('lengths of fcivec ',len(h_eff_energies), len(h_eff_fcivec[0]))

    descriptors = {}
    descriptors["energy"] = h_eff_energies
    for k, it in onebody.items():
        # print(it.shape, rdm1.shape)
        descriptors[k] = np.einsum("mn,ismn -> i", it, rdm1)
    for k, it in twobody.items():
        descriptors[k] = np.einsum(
            "mnop,ismnop -> i", it, rdm2[:, [0, 1, 3], :, :, :, :]
        )
    return descriptors, h_eff_fcivec


# new distance matrix bit:
def norm_distance_matrix(
    ai_df: pd.DataFrame, model_descriptors: dict, matches: list[str], scaler=None
) -> np.array:
    """
    Inputs:
        matches : List of descriptor names as strings
    Could include in the future, weights : the weight of each descriptor.

    Returns:
        nstates_ai by nstates_model estimation of the distance
    """
    nai = len(ai_df)
    nmodel = model_descriptors[matches[0]].shape[0]
    distance = np.zeros((nai, nmodel))

    for k in matches:
        scaler_model = scaler.fit(ai_df[k].values.reshape(-1, 1))  # If scaler present
        xai = scaler_model.transform(ai_df[k].values.reshape(-1, 1))[
            :, 0
        ]  # If scaler present
        # xai = ai_df[k].values # No scaler
        scaler_model = scaler.fit(
            model_descriptors[k].reshape(-1, 1)
        )  # If scaler present
        xmodel = scaler_model.transform(model_descriptors[k].reshape(-1, 1))[
            :, 0
        ]  # If scaler present
        # xmodel = (model_descriptors[k]) # If no scaler
        distance += (xai[:, np.newaxis] - xmodel[np.newaxis, :]) ** 2  # euclidean
    return distance


def descriptor_distance(
    ai_df: pd.DataFrame, model_descriptors: dict, matches: list[str], scaler, norm
) -> np.array:  # all descriptors are normalized together from 0 to 1
    """
    Inputs:
        matches : List of descriptor names as strings
        scaler  : from scipy.preprocessing, default MinMaxScaler() which scales input from 0 to 1
    Could include in the future, weights : the weight of each descriptor.

    Returns:
        nstates_ai by nstates_model estimation of the distance
    """
    nai = len(ai_df)
    nmodel = model_descriptors[matches[0]].shape[0]
    distance = np.zeros((nai, nmodel))

    for k in matches:
        xai = ai_df[k].values
        xmodel = model_descriptors[k]

        distance += (xai[:, np.newaxis] - xmodel[np.newaxis, :]) ** 2 / norm[
            k
        ]  # euclidean

    distance = distance / len(matches)

    # scaler_distance = scaler.fit(distance.reshape(-1, 1))            # finds the scale transform

    # distance = scaler_distance.transform(distance.reshape(-1, 1))    # transforms the input to 0 to 1
    # distance = distance.reshape((nai, nmodel))                       # Return to original shape
    return distance


def distance_matrix(
    ai_df: pd.DataFrame, model_descriptors: dict, matches: list[str]
) -> np.array:
    """
    Inputs:
        matches : List of descriptor names as strings
    Could include in the future, weights : the weight of each descriptor.

    Returns:
        nstates_ai by nstates_model estimation of the distance
    """
    nai = len(ai_df)
    nmodel = model_descriptors[matches[0]].shape[0]
    distance = np.zeros((nai, nmodel))

    for k in matches:
        xai = ai_df[k].values
        xmodel = model_descriptors[k]
        distance += (xai[:, np.newaxis] - xmodel[np.newaxis, :]) ** 2  # euclidean
    return distance


def spectrum_loss(energy_ab, energy_m_mapped):
    sloss = np.sum((energy_ab - energy_m_mapped) ** 2) / energy_ab.shape[0]  # MSE
    return sloss


def spectrum_loss_abs(energy_ab, energy_m_mapped):
    sloss = (
        np.sum(np.abs(energy_ab - energy_m_mapped)) / energy_ab.shape[0]
    )  # Abs difference # Don't think I need this, this is an unusual metric to use
    return sloss


def descriptor_distance_loss(distance, row_ind, col_ind, N):
    dloss = np.sum(distance[row_ind, col_ind]) / N
    return dloss


def unmapped_penalty(energy_m_unmapped, max_energy_ab: float, norm):
    penalty = np.sum(
        (np.maximum(0, max_energy_ab - energy_m_unmapped) ** 2) / norm["energy"]
    )
    return penalty


def rotation_from_angles(thetas, N):
    inds = np.triu_indices(N, k=1)

    T = np.zeros((N, N))
    T[inds[0], inds[1]] = thetas
    T = T - T.T
    return scipy.linalg.expm(T)


# ^ Fix later, need to apply mask to the resulting array for specific states


def temp_from_beta(beta):
    # beta units Ha, the energy unit for the ab initio
    k_b = 3.166811563e-6  # Units Ha/K
    temp = 1 / k_b * beta
    return temp  # K


def give_boltzmann_weights(energy_ab, beta):
    # give boltzmann weights given the boltzmann factor
    # beta = 1/(k_b * T)

    e_i = energy_ab[0]
    boltzmann_weights = np.exp(-beta * (energy_ab - e_i))
    return boltzmann_weights


def plot_distance(distance, name="test_dist.png", label="distance"):

    plt.imshow(distance)
    plt.ylabel("ab initio #")
    plt.xlabel("model #")
    plt.colorbar(label=label)
    plt.savefig(name)
    plt.close()

    return


def evaluate_loss_overall(
    params: np.ndarray,
    keys,
    weights,
    boltzmann_weights,
    onebody,
    twobody,
    ai_df,
    nroots,
    matches,
    scaler,
    fcivec,
    norm,
) -> float:

    w_0 = weights[0]
    w_1 = weights[1]
    lamb = weights[2]

    if fcivec == None:
        fcivec = cache["fcivec"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )
    cache["fcivec"] = fcivec

    dist_des = descriptor_distance(
        ai_df, descriptors, matches=matches, scaler=scaler, norm=norm
    )
    dist_energy = descriptor_distance(
        ai_df, descriptors, matches=["energy"], scaler=scaler, norm=norm
    )
    distance = dist_energy + dist_des
    row_ind, col_ind = linear_sum_assignment(distance)

    sloss = np.sum(boltzmann_weights * dist_energy[row_ind, col_ind])
    dloss = np.sum(boltzmann_weights * dist_des[row_ind, col_ind])
    not_col_ind = np.delete(np.arange(nroots), col_ind)
    penalty = unmapped_penalty(
        descriptors["energy"][not_col_ind], np.max(ai_df["energy"]), norm=norm
    )

    loss = (
        w_0 * sloss + w_1 * dloss + w_0 * (penalty**2) + lamb * np.sum(np.abs(params))
    )
    # print("Loss: ", loss)

    return loss


def mapping(
    onebody: dict,
    twobody: dict,
    onebody_params: list,
    twobody_params: list,
    ai_df: pd.DataFrame,
    nroots: int,
    outfile: str,
    matches: list,
    weights: list,
    beta: float,
    guess_params=None,
):
    """

    Inputs: 
    onebody: dict (keys are strings, values are 2D np.arrays) tij
    twobody: dict (keys are strings, values are 4D np.arrays) Vijkl
    onebody_params: list of strings, keys that are allowed to have nonzero values in the Hamiltonian
    twobody_params: list of strings, keys that are allowed to have nonzero values in the Hamiltonian
    ai_df: pd.DataFrame, ab initio data, columns are parameter names, and 'energy'
    nroots: int, number of roots to solve for in the model
    outfile: str, output to HDF5 file
    matches: list of strings, keys that are used to match the ab initio and model descriptors
    weights: list of floats, [w_0, w_1, lambda] weights for the spectrum loss, descriptor loss, and lasso penalty
    beta: float, inverse temperature for the boltzmann weights. 0 means equal weights to all states
    guess_params: dict (keys are strings, values are floats) overrides the dmd parameters

    For nroots, we want to have more than were done in ab initio. The model roots will get mapped onto the ab initio ones, with
    the extra ones dropped. As the number of roots increases, the mapping should get a bit better.
    """
    dmd = sm.OLS(ai_df["energy"], ai_df[onebody_params + twobody_params]).fit()
    print(dmd.summary())

    print("Intial Solve, might take a while")
    params = dmd.params.copy()

    if guess_params is not None:
        for k in guess_params.keys():
            params[k] = guess_params[k]

    print("Starting Parameters: ", params)

    descriptors, fcivec = solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots
    )
    cache["fcivec"] = fcivec

    print("Ab initio df shape: ", ai_df.shape)

    scaler = MinMaxScaler()
    norm = 2 * np.var(ai_df)

    dist_des = descriptor_distance(
        ai_df, descriptors, matches=matches, scaler=scaler, norm=norm
    )
    dist_energy = descriptor_distance(
        ai_df, descriptors, matches=["energy"], scaler=scaler, norm=norm
    )
    distance = dist_energy + dist_des
    plot_distance(distance)
    row_ind, col_ind = linear_sum_assignment(distance)

    boltzmann_weights = give_boltzmann_weights(ai_df["energy"], beta)
    print(f"Boltzmann weights for beta {beta}: {boltzmann_weights}")

    # x = np.zeros_like(distance)
    # x[row_ind, col_ind] = 1
    # plot_distance(x, name="mapping_test.png", label="mapping")

    # Save data to look at optmization
    list_params = []
    list_loss = []
    list_sloss = []
    list_dloss = []

    N = len(matches)

    def evaluate_loss(rs: np.ndarray, keys, weights, boltzmann_weights) -> float:

        w_0 = weights[0]
        w_1 = weights[1]
        lamb = weights[2]

        fcivec = cache["fcivec"]

        params = pd.Series(rs, index=keys)

        descriptors, fcivec = solve_effective_hamiltonian(
            onebody, twobody, params, nroots=nroots, ci0=fcivec
        )
        cache["fcivec"] = fcivec

        dist_des = descriptor_distance(
            ai_df, descriptors, matches=matches, scaler=scaler, norm=norm
        )
        dist_energy = descriptor_distance(
            ai_df, descriptors, matches=["energy"], scaler=scaler, norm=norm
        )
        distance = dist_energy + dist_des
        row_ind, col_ind = linear_sum_assignment(distance)

        sloss = np.sum(boltzmann_weights * dist_energy[row_ind, col_ind])
        dloss = np.sum(boltzmann_weights * dist_des[row_ind, col_ind])
        not_col_ind = np.delete(np.arange(nroots), col_ind)
        penalty = unmapped_penalty(
            descriptors["energy"][not_col_ind], np.max(ai_df["energy"]), norm=norm
        )

        loss = w_0 * sloss + w_1 * dloss + w_0 * (penalty)  # + lamb*penalty
        print("Loss: ", loss)

        list_params.append(rs)
        list_loss.append(loss)
        list_sloss.append(sloss)
        list_dloss.append(dloss)

        return loss

    def callback(x):
        # print(fun)
        pass

    # print( "Spectrum Loss (Ha)" , spectrum_loss(ai_df['energy'], descriptors['energy'][col_ind]) )
    # print( "Descriptor Loss " , descriptor_distance_loss(distance, row_ind, col_ind, N=len(matches) ) )
    # not_col_ind = np.delete(np.arange(nroots), col_ind)
    # print( "Unmapped states penalty (Ha)", unmapped_penalty(descriptors['energy'][not_col_ind], np.max(ai_df['energy'])) )
    # exit()

    keys = params.keys()
    x0 = params.values

    # OPTMIZATION LOOP START
    print("Starting optimization")
    # e_tol = 0.05 # We expect the parameters to be within 0.05 Ha of the correct parameter
    # bnds = ( (x0[0]-e_tol, x0[0]+e_tol), (x0[1]-2*e_tol, 0), (x0[2]-2*e_tol,x0[2]+e_tol) ) #E0, t, U bounds min max : E0 is neg, U is pos, t is neg
    # print("Bounds for paramters E0, U, t:", bnds)

    """ Nelder-Mead Optimize"""
    # xmin = minimize(evaluate_loss, x0, args=(keys, weights), method='Nelder-Mead', tol=1e-7, callback = callback, options={'maxiter': 100})

    """ L-BFGS-B Optimize"""
    # xmin = minimize(evaluate_loss, x0, args=(keys, weights), method='L-BFGS-B', tol=1e-7, bounds=bnds, callback = callback, options={'maxiter': 100})

    """ BFGS Optimize"""
    xmin = minimize(
        evaluate_loss_overall,
        x0,
        args=(
            keys,
            weights,
            boltzmann_weights,
            onebody,
            twobody,
            ai_df,
            nroots,
            matches,
            scaler,
            None,
            norm,
        ),
        method="BFGS",
        tol=1e-7,
        callback=callback,
        options={"maxiter": 1000},
    )
    # vs
    # xmin = minimize(evaluate_loss, x0, args=(keys, weights, boltzmann_weights), method='BFGS', tol=1e-7, callback = callback, options={'maxiter': 1000})

    """ Basin Hopping Optimize"""
    # minimizer_kwargs = {"method": "BFGS", "args":(keys, weights, boltzmann_weights, onebody, twobody, ai_df, nroots, matches, scaler, None)}
    # xmin = basinhopping(evaluate_loss_overall, x0, minimizer_kwargs=minimizer_kwargs, niter=10, seed=10, T=0.05)

    print(xmin.nit, xmin.nfev)
    print(xmin.message)

    # ENDING OPTIMIZATION ANALYSIS
    params = pd.Series(xmin.x, index=keys)

    # get loss, sloss, dloss, penalty saved too
    fcivec = cache["fcivec"]
    descriptors, fcivec = solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )

    dist_des = descriptor_distance(
        ai_df, descriptors, matches=matches, scaler=scaler, norm=norm
    )
    dist_energy = descriptor_distance(
        ai_df, descriptors, matches=["energy"], scaler=scaler, norm=norm
    )
    distance = dist_energy + dist_des
    row_ind, col_ind = linear_sum_assignment(distance)

    sloss_MSE = spectrum_loss(ai_df["energy"], descriptors["energy"][col_ind])
    sloss_abs_diff = spectrum_loss_abs(ai_df["energy"], descriptors["energy"][col_ind])
    # dloss = descriptor_distance_loss(distance, row_ind, col_ind, N=N )
    not_col_ind = np.delete(np.arange(nroots), col_ind)
    penalty = unmapped_penalty(
        descriptors["energy"][not_col_ind], np.max(ai_df["energy"]), norm=norm
    )
    # ^ The above code is the same as the evaluate_loss function.

    with h5py.File(outfile, "w") as f:
        for k in descriptors:
            f["descriptors/" + k] = descriptors[k][...]
        for k in onebody_params + twobody_params:
            f["dmd_params/" + k] = dmd.params[k]
        for k in onebody_params + twobody_params:
            f["rdmd_params/" + k] = params[k]
        f["association"] = col_ind
        f["para_w_0"] = weights[0]
        f["para_w_1"] = weights[1]
        f["para_lamb"] = weights[2]
        f["loss_loss"] = xmin.fun
        f["ai_spectrum_range (Ha)"] = np.max(ai_df["energy"]) - np.min(ai_df["energy"])

        # Clean up saved attributes
        f["loss_spectrum_MSE_Ha"] = sloss_MSE
        f["loss_spectrum_RMSE_Ha"] = np.sqrt(sloss_MSE)
        f["loss_spectrum_abs_diff_Ha"] = (
            sloss_abs_diff  # I don't think this needs to be saved
        )
        f["state_loss_norm_spectrum"] = dist_energy[row_ind, col_ind]
        f["state_loss_norm_distance"] = dist_des[row_ind, col_ind]
        f["loss_norm_spectrum"] = np.sum(
            boltzmann_weights * dist_energy[row_ind, col_ind]
        )
        f["loss_norm_distance"] = np.sum(boltzmann_weights * dist_des[row_ind, col_ind])
        f["loss_distance"] = np.sum(
            distance_matrix(ai_df, descriptors, matches=matches)[row_ind, col_ind]
        )
        f["loss_penalty_Ha"] = penalty
        f["loss_lasso"] = lamb * np.sum(np.abs(params))

        f["iterations"] = xmin.nit
        f["iter_params"] = np.array(list_params)
        f["iter_loss"] = np.array(list_loss)
        f["iter_sloss"] = np.array(list_sloss)
        f["iter_dloss"] = np.array(list_dloss)
        f["termination_message"] = xmin.message


def make_mapping(
    named_terms,
    ai_descriptors,
    model_descriptors,
    nroots,
    onebody_params,
    twobody_params,
    minimum_1s_occupation=3.7,
    w_0=1,
    beta=0,
    lamb=0,
    guess_params=None,
):
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
    ai_df = pd.read_csv(ai_descriptors)
    ai_df = ai_df[ai_df.E0 > minimum_1s_occupation]
    ai_df = ai_df[
        ai_df.U < 1.3
    ]  # Need to remove the top two states from the optimization
    # df=df[:30]

    # print(df.shape)

    matches = onebody_keys[1:] + twobody_keys  # ['t', 'U' , 'ni_hop'] #'V'

    w_1 = 1 - w_0

    weights = [w_0, w_1, lamb]  # w_0, w_1, lamb

    mapping(
        onebody,
        twobody,
        onebody_params,
        twobody_params,
        ai_df,
        nroots,
        model_descriptors,
        matches,
        weights,
        beta,
        guess_params,
    )


if __name__ == "__main__":
    # r = 3.6
    # rs = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0]
    rs = [5.0]
    # rs = [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0]
    # rs = [7.0] #[3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0] # redo these two

    betas = [4.0]  # 0.0, 0.5, 2, 4, 8,
    lamb = 128

    model_names = []
    parameters = []

    onebody_params = ["E0", "t"]
    twobody_params = ["U"]  # ['U', 'V']

    model_name = ""
    for k in onebody_params + twobody_params:
        model_name += f"-{k}"
    print(model_name)
    model_names.append(model_name)
    parameters.append([onebody_params, twobody_params])

    """ onebody_params = ['E0']
    twobody_params = ['U', 'ni_hop', 'V']

    model_name = ''
    for k in onebody_params+twobody_params:
        model_name += f'-{k}'
    print(model_name)
    model_names.append(model_name)
    print(model_names)
    parameters.append([onebody_params,twobody_params]) """

    # w_0s = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 0.99, 0.995, 1.0])
    # w_0s = np.array([0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 0.99, 0.995, 1.0])
    # w_0s = np.array([0.95, 0.99, 0.995, 1.0])

    w_0s = [0.6]

    df_guess = pd.read_csv("guess_parameters.csv", index_col=False)
    # guess_params = {'E0':-0.47, # Given good guess after renormalization runs from dmd
    #                 't':-0.099,
    #'U':0.25, #'U':0.36162820370768256,  # Use DMD U
    #                 'ni_hop':-0.098}
    #                 #'ni_hop2':-0.007640999400566293,
    #                 #'V':0.03468826370502873} #'V':0.01}

    for beta in betas:
        for i, model_name in enumerate(model_names):
            for r in rs:
                for w_0 in w_0s:
                    onebody_params = parameters[i][0]
                    twobody_params = parameters[i][1]
                    print("r ", r)
                    print("w0 ", w_0)
                    print(onebody_params, twobody_params)

                    dir = f"r{r}/{model_name}"
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    # for key in guess_params.keys():
                    #    guess_params[key] = df_guess[df_guess['r'] == r].to_dict('r')[0][key] # Guess Parameters

                    data_dir = f"../../../sonali_H4_paper/3_data/h4_square_mol/data/r{r}/"  # f"../test_data/r{r}/"
                    dfile = f"r{r}/ai_descriptors.csv"

                    model_file = f"{dir}/model_descriptors_r{r}_{model_name}_w0{w_0}_beta{beta}_lamb{lamb}.hdf5"

                    if os.path.isfile(model_file):
                        continue

                    make_mapping(
                        "named_terms_new.hdf5",
                        dfile,
                        model_file,
                        36,
                        onebody_params,
                        twobody_params,
                        minimum_1s_occupation=3.7,
                        w_0=w_0,
                        beta=beta,
                        lamb=lamb,
                    )  # , guess_params=guess_params)
