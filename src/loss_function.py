import pandas as pd
import statsmodels.api as sm
import h5py
import numpy as np
from rich import print
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.preprocessing import MinMaxScaler
import solver


eV2Ha = 0.0367492929
cache = {}  # For speedup during mulitiple ED during model optimization


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


def unmapped_penalty(energy_m_unmapped, max_energy_ab: float, norm):
    penalty = np.sum(
        (np.maximum(0, max_energy_ab - energy_m_unmapped) ** 2) / norm["energy"]
    )
    return penalty


def give_boltzmann_weights(energy_ab, beta):
    # give boltzmann weights given the boltzmann factor
    # beta = 1/(k_b * T)

    e_i = energy_ab[0]
    boltzmann_weights = np.exp(-beta * (energy_ab - e_i))
    return boltzmann_weights


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

    if fcivec is None and "fcivec" in cache:
        fcivec = cache["fcivec"]

    params = pd.Series(params, index=keys)
    print(params)
    descriptors, fcivec = solver.solve_effective_hamiltonian(
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

    return {
        "loss": loss,
        "sloss": sloss,
        "dloss": dloss,
        "penalty": penalty,
        "descriptors": descriptors,
        "distance": distance,
        "row_ind": row_ind,
        "col_ind": col_ind,
        "params": params,
    }


def optimize_function(*args, **kwargs):
    return evaluate_loss_overall(*args, **kwargs)["loss"]


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

    print("Initial Solve, might take a while")
    params = dmd.params.copy()

    if guess_params is not None:
        for k in guess_params.keys():
            params[k] = guess_params[k]

    print("Starting Parameters: ", params)

    scaler = MinMaxScaler()
    norm = 2 * np.var(ai_df)

    boltzmann_weights = give_boltzmann_weights(ai_df["energy"], beta)
    print(f"Boltzmann weights for beta {beta}: {boltzmann_weights}")

    keys = params.keys()
    x0 = params.values

    # OPTMIZATION LOOP START
    print("Starting optimization")

    xmin = minimize(
        optimize_function,
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
        jac="3-point",
        method="Powell",
        tol=1e-7,
        options={"maxiter": 1000},
    )

    print(xmin.nit, xmin.nfev)
    print(xmin.message)
    print("function value", xmin.fun)
    print("parameters", xmin.x)

    data = evaluate_loss_overall(
        xmin.x,
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
    )

    with h5py.File(outfile, "w") as f:
        for k in onebody_params + twobody_params:
            f["dmd_params/" + k] = dmd.params[k]
        for k in onebody_params + twobody_params:
            f["rdmd_params/" + k] = params[k]
        f["para_w_0"] = weights[0]
        f["para_w_1"] = weights[1]
        f["para_lamb"] = weights[2]
        f["loss_loss"] = xmin.fun
        f["ai_spectrum_range (Ha)"] = np.max(ai_df["energy"]) - np.min(ai_df["energy"])

        for k in data:
            if k == "descriptors":
                for kk in data[k]:
                    f[k + "/" + kk] = data[k][kk]
            else:
                f[k] = data[k]

        f["iterations"] = xmin.nit
        f["termination_message"] = xmin.message
