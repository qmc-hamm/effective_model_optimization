import pandas as pd
import statsmodels.api as sm
import h5py
import numpy as np
from rich import print
from scipy.optimize import linear_sum_assignment, minimize
import solver

cache = {}  # For speedup during mulitiple ED during model optimization


def descriptor_distance(
    ai_df: pd.DataFrame, model_descriptors: dict, matches: list[str], norm
) -> np.array:  # all descriptors are normalized together from 0 to 1
    """Calculates the distance of the descriptors after normalizing, given the list of matching parameters.
    Formula: 

    Parameters
    ----------
    ai_df : pd.DataFrame
        The descriptor information of the ab initio data. The keys should match model descriptors'.
    model_descriptors : dict
        The descriptor information of the model data. Normally is generated by solver.py.
    matches : list[str]
        List of descriptor names as strings
    norm : float
        Norm to normalize the incoming data. (sigma_{E}_{ab})^2

    Returns
    -------
    np.array
        nstates_ai by nstates_model estimation of the distance
    """
    nai = len(ai_df)
    nmodel = model_descriptors[matches[0]].shape[0]
    distance = np.zeros((nai, nmodel))

    for k in matches:
        xai = ai_df[k].values
        xmodel = model_descriptors[k]

        distance += (xai[:, np.newaxis] - xmodel[np.newaxis, :]) ** 2 / norm[k]  # euclidean

    distance = distance / len(matches)
    return distance


def unmapped_penalty(energy_m_unmapped, max_energy_ab: float, norm):
    """Fills out the rest of the distance matrix. The penalty matrix to map unmapped to ab initio states. 

    Parameters
    ----------
    energy_m_unmapped : nd.array(float)
        The energies of the model states that are unmapped.
    max_energy_ab : float
        Maximum ab initio energy in the dataset.
    norm : float
        Norm to normalize the incoming data. (sigma_{E}_{ab})^2

    Returns
    -------
    nd.array
        Shape: (len(energy_m_unmapped), nroots for model). Penalty for the unmapped states to be mapped to 
        ab initio states.
    """
    penalty = np.sum(
        (np.maximum(0, max_energy_ab - energy_m_unmapped) ** 2) / norm["energy"]
    )
    return penalty


def give_boltzmann_weights(energy_ab, e_0, beta):
    """Return the boltzmann weights given a beta

    Parameters
    ----------
    energy_ab : nd.array(float)
        Length of ab initio states, the energy of each state.
    e_0 : float
        The ground state energy
    beta : float
        Boltmann beta, sets the expentional decay of the state weighting.

    Returns
    -------
    nd.array(float)
        boltztman weights, the ground state always get 1. lower energy states get more weighting then exponentially 
        decays.
    """    
    # give boltzmann weights given the boltzmann factor
    # beta = 1/(k_b * T)

    boltzmann_weights = np.exp(-beta * (energy_ab - e_0))
    return boltzmann_weights


def evaluate_loss(
    params: np.ndarray,
    keys,
    weights,
    boltzmann_weights,
    onebody,
    twobody,
    ai_df,
    max_ai_energy,
    nroots,
    matches,
    fcivec,
    norm,
) -> float:
    """The actual Loss functional.

    Parameters
    ----------
    params : np.ndarray
        Parameters of the hamiltonian. The keys should match the key names of the onebody and twobody operators. 
    keys : List
        Used to make params a pd.Series. keys should match the key names of the onebody and twobody operators.
    weights : List
        w_0, w_1, lambda ; lambda not used -> could be used for lasso
    boltzmann_weights : nd.array(float)
        boltztman weights, the ground state always get 1. lower energy states get more weighting then exponentially 
        decays.
    onebody : dict
        (keys are strings, values are 2D np.arrays) tij. Dictionary of the 1-body operators. Each operator of the 
        shape (norb,norb).
    twobody : dict
        (keys are strings, values are 4D np.arrays) Vijkl. Dictionary of the 2-body operators. Each operator of the 
        shape (norb,norb,norb,norb).
    ai_df : pd.DataFrame
        The descriptor information of the ab initio data. The keys should match model descriptors'.
    max_ai_energy : float
        Maximum of the whole ab initio data. See np.max(ai_df["energy"]) in mapping.
    nroots : int
        Number of eigenstates to solve the model Hamiltonian for. 
    matches : list[str]
        List of descriptor names as strings
    fcivec : nd.array()
        Stored fcivecs to pass into the solver, so that full model diagonalization does not take as long during 
        optimization.
    norm : float
        Norm to normalize the incoming data. (sigma_{E}_{ab})^2

    Returns
    -------
    float
        Loss function value.
    """
    w_0 = weights[0]
    w_1 = weights[1]
    lamb = weights[2]

    if fcivec is None and "fcivec" in cache:
        fcivec = cache["fcivec"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )
    cache["fcivec"] = fcivec

    dist_des = descriptor_distance(
        ai_df, descriptors, matches=matches, norm=norm
    )
    dist_energy = descriptor_distance(
        ai_df, descriptors, matches=["energy"], norm=norm
    )
    distance = dist_energy + dist_des
    row_ind, col_ind = linear_sum_assignment(distance)

    sloss = np.sum(boltzmann_weights * dist_energy[row_ind, col_ind])
    dloss = np.sum(boltzmann_weights * dist_des[row_ind, col_ind])
    not_col_ind = np.delete(np.arange(nroots), col_ind)

    # Debugging information to test penalty 
    #print(row_ind)
    #print(col_ind)
    #print()
    #print(not_col_ind)
    #print(descriptors["energy"])
    penalty = unmapped_penalty(
        descriptors["energy"][not_col_ind], max_ai_energy, norm=norm
    )

    loss = (
        w_0 * sloss + w_1 * dloss #+ w_0 * (penalty)
    )

    # Debugging information to test loss
    #print()
    #print("loss ", loss)
    #print("params", params.values)

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
    return evaluate_loss(*args, **kwargs)["loss"]


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
    p: int,
    guess_params=None,
):
    """Sets up the loss function to then optimize.

    For nroots, we want to have more than were done in ab initio. The model roots will get mapped onto the ab initio ones, with
    the extra ones dropped. As the number of roots increases, the mapping should get a bit better.

    Parameters
    ----------
    onebody : dict
        (keys are strings, values are 2D np.arrays) tij. Dictionary of the 1-body operators. Each operator of the 
        shape (norb,norb).
    twobody : dict
        (keys are strings, values are 4D np.arrays) Vijkl. Dictionary of the 2-body operators. Each operator of the 
        shape (norb,norb,norb,norb).
    onebody_params : list
        List of onebody keys which are the parameters to set inside the hamiltonian. list of strings, keys that are 
        allowed to have nonzero values in the Hamiltonian
    twobody_params : list
        List of twobody keys which are the parameters to set inside the hamiltonian. list of strings, keys that are 
        allowed to have nonzero values in the Hamiltonian
    ai_df : pd.DataFrame
        The descriptor information of the ab initio data. The keys should match model descriptors'. ab initio data, columns are parameter names, and 'energy'
    nroots : int
        Number of eigenstates to solve the model Hamiltonian for. number of roots to solve for in the model
    outfile : str
        File to save the model file to. output to HDF5 file
    matches : list[str]
        List of descriptor names as strings. keys that are used to match the ab initio and model descriptors. 
        Should be the descriptors corresponding to the terms in the hamiltonian.
    weights : list
        w_0, w_1, lambda. list of floats, [w_0, w_1, lambda] weights for the spectrum loss, descriptor loss, 
        and lasso penalty
    beta : float
        inverse temperature for the boltzmann weights. 0 means equal weights to all states
    p : int
        Number of states to drop out of optimization for CV.
    guess_params : pd.Series
        Guess parameters to override the DMD parameters if so desired. dict (keys are strings, values are floats) 
        overrides the dmd parameters
    """
    p_out_states = np.random.choice(np.arange(0, len(ai_df)),size=p, replace=False)
    #p_out_states = np.array([p]) # this p is for 30-k_groups, leaving out data 1-by-1
    ai_df_train =  ai_df.drop(p_out_states, axis=0)
    ai_df_test =  ai_df.loc[p_out_states]
    max_ai_energy = np.max(ai_df["energy"])

    boltzmann_weights_train = give_boltzmann_weights(ai_df_train["energy"], ai_df["energy"][0], beta)
    print(f"Boltzmann weights for beta {beta}: {boltzmann_weights_train}")
    boltzmann_weights_test = give_boltzmann_weights(ai_df_test["energy"], ai_df["energy"][0], beta)
    print(f"Boltzmann weights for beta {beta}: {boltzmann_weights_test}")

    # Starting parameter guess, not sure if this should stay the same despite the p split
    dmd = sm.OLS(ai_df["energy"], ai_df[onebody_params + twobody_params]).fit()
    print(dmd.summary())

    params = dmd.params.copy()

    if guess_params is not None:
        for k in guess_params.keys():
            params[k] = guess_params[k]

    print("Starting Parameters: ", params)

    norm = 2 * np.var(ai_df) # Should this be based on the ai_df_train? I think not.

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
            boltzmann_weights_train,
            onebody,
            twobody,
            ai_df_train,
            max_ai_energy,
            nroots,
            matches,
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

    print("Evaluate train data after optimization:")
    data_train = evaluate_loss(
        xmin.x,
        keys,
        weights,
        boltzmann_weights_train,
        onebody,
        twobody,
        ai_df_train,
        max_ai_energy,
        nroots,
        matches,
        None,
        norm,
    )

    N = len(ai_df_train)
    print("loss per state :", data_train['loss']/N)
    print("Spectrum loss per state  :", data_train['sloss']/N)
    print("Descriptor loss per state:", data_train['dloss']/N)
    print("penalty:", data_train['penalty'])


    data_test = evaluate_loss(
        xmin.x,
        keys,
        weights,
        boltzmann_weights_test,
        onebody,
        twobody,
        ai_df_test,
        max_ai_energy,
        nroots,
        matches,
        None,
        norm,
    )

    N = len(ai_df_test)
    print("Test: loss per state :", data_test['loss']/N)
    print("Test: Spectrum loss per state  :", data_test['sloss']/N)
    print("Test: Descriptor loss per state:", data_test['dloss']/N)
    print("Test: penalty:", data_test['penalty'])

    with h5py.File(outfile, "w") as f:
        for k in onebody_params + twobody_params:
            f["dmd_params/" + k] = dmd.params[k]
        f["para_w_0"] = weights[0]
        f["para_w_1"] = weights[1]
        f["para_lamb"] = weights[2]
        f["loss_loss"] = xmin.fun
        f["ai_spectrum_range (Ha)"] = np.max(ai_df["energy"]) - np.min(ai_df["energy"])

        f["nstates_train"] = len(ai_df_train)
        f["nstates_test"]  = len(ai_df_test)
        f["state_ind_for_test"] = p_out_states

        for k in data_train:
            if k == "descriptors":
                for kk in data_train[k]:
                    f["train/"+ k + "/" + kk] = data_train[k][kk]
            elif k == "params":
                for i, kk in enumerate(onebody_params + twobody_params):
                    f["train/"+"rdmd_params/" + kk] = data_train[k][i]
            else:
                f["train/"+k] = data_train[k]

        for k in data_test:
            if k == "descriptors":
                for kk in data_test[k]:
                    f["test/"+ k + "/" + kk] = data_test[k][kk]
            elif k == "params":
                for i, kk in enumerate(onebody_params + twobody_params):
                    f["test/"+"rdmd_params/" + kk] = data_test[k][i]
            else:
                f["test/"+k] = data_test[k]

        f["iterations"] = xmin.nit
        f["termination_message"] = xmin.message
