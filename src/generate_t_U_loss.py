import h5py
import pandas as pd
import loss_function
import numpy as np
from scipy.optimize import linear_sum_assignment
import solver

cache = {}


def return_set_up_h4(
    named_terms,
    ai_descriptors,
    model_descriptors,
    nroots,
    onebody_params,
    twobody_params,
    minimum_1s_occupation=3.7,
    w_0=1,
    beta=0,
):
    """Set up H4 molecule as an example.

    Parameters
    ----------
    named_terms : str
        File name for symmetry invariant terms.
    ai_descriptors : pd.DataFrame
        The descriptor information of the ab initio data. The keys should match model descriptors'. ab initio data, columns are parameter names, and 'energy'
    model_descriptors : str
        Which output file to save the model data to.
    nroots : int
        Number of eigenstates to solve the model Hamiltonian for. number of roots to solve for in the model
    onebody_params : list
        List of onebody keys which are the parameters to set inside the hamiltonian. list of strings, keys that are 
        allowed to have nonzero values in the Hamiltonian
    twobody_params : list
        List of twobody keys which are the parameters to set inside the hamiltonian. list of strings, keys that are 
        allowed to have nonzero values in the Hamiltonian
    minimum_1s_occupation : float, optional
        Sets the cut of the ai_descriptors to only include states with this minimum 1s occupancy, by default 3.7
    w_0 : int, optional
        Weighting of the spectrum loss. The descriptor loss is weighted as 1-w0. by default 1
    beta : float, optional
        inverse temperature for the boltzmann weights. 0 means equal weights to all states, by default 0
    """
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
    ai_df = ai_df[ai_df.U < 1.3]  # Need to remove the top two states from the optimization

    matches = onebody_params + twobody_params

    w_1 = 1 - w_0

    weights = [w_0, w_1, 0.0]  # w_0, w_1, lamb

    return ai_df, beta, nroots, weights, onebody, twobody, matches


def generate_t_U(tmin: float,
                 tmax: float,
                 Umin: float,
                 Umax: float,
                 eps: float = -0.5,
                 ngrid: int = 10,
                 outfile: str = "loss_grid.csv"):
    """_summary_

    Parameters
    ----------
    tmin : float
        Starting t range. t is the hopping parameter.
    tmax : float
        Ending t range.
    Umin : float
        Starting U range. U is the double occupancy parameter.
    Umax : float
        Ending U range.
    eps : float, optional
        E0 value or site/orbital energy., by default -0.5
    ngrid : int, optional
        How many grid points in 1D, by default 10
    outfile : str, optional
        File name to save to, by default "loss_grid.csv"
    """

    onebody_params = ["E0", "t"]
    twobody_params = ["U"]  
    keys = onebody_params + twobody_params

    set_up = return_set_up_h4("../h4_data/named_terms_new.hdf5",
                              "../h4_data/ai_descriptors.csv",
                              "model_output.hdf5",
                              nroots=36,  # (4 sites choose 2 electrons)^2 , ^2 is for both spin up and spin down
                              onebody_params=onebody_params,
                              twobody_params=twobody_params,
                              minimum_1s_occupation=3.7,
                              w_0=0.7,
                              beta=5.0,
                              )

    ai_df, beta, nroots, weights, onebody, twobody, matches = set_up

    norm = 2 * np.var(ai_df)
    max_ai_energy = max(ai_df["energy"])
    boltzmann_weights = loss_function.give_boltzmann_weights(ai_df["energy"], ai_df["energy"][0], beta)

    loss_grid = []
    for t in np.linspace(tmin, tmax, num=ngrid):
        print(f"t {t}")
        for U in np.linspace(Umin, Umax, num=ngrid):

            params = [eps, t, U]  # loop through these with same ordering as keys
            
            data_sidb = evaluate_loss_sidb(
                params,
                keys,
                weights,
                boltzmann_weights,
                onebody,
                twobody,
                ai_df,
                max_ai_energy,
                nroots,
                matches,
                None,
                norm,
            )

            data_sid = evaluate_loss_sid(
                params,
                keys,
                weights,
                boltzmann_weights,
                onebody,
                twobody,
                ai_df,
                max_ai_energy,
                nroots,
                matches,
                None,
                norm,
            )

            data_si = evaluate_loss_si(
                params,
                keys,
                weights,
                boltzmann_weights,
                onebody,
                twobody,
                ai_df,
                max_ai_energy,
                nroots,
                matches,
                None,
                norm,
            )

            data_sd = evaluate_loss_sd(
                params,
                keys,
                weights,
                boltzmann_weights,
                onebody,
                twobody,
                ai_df,
                max_ai_energy,
                nroots,
                matches,
                None,
                norm,
            )

            data_s = evaluate_loss_s(
                params,
                keys,
                weights,
                boltzmann_weights,
                onebody,
                twobody,
                ai_df,
                max_ai_energy,
                nroots,
                matches,
                None,
                norm,
            )

            new_data = {}
            new_data["loss_sidb"] = data_sidb["loss"]
            new_data["loss_sid"] = data_sid["loss"]
            new_data["loss_si"] = data_si["loss"]
            new_data["loss_sd"] = data_sd["loss"]
            new_data["loss_s"] = data_s["loss"]

            for i, k in enumerate(keys):
                new_data[k] = data_sidb["params"][i]

            loss_grid.append(new_data)

    df = pd.DataFrame(loss_grid)
    df.to_csv(outfile)

    return


# s = spectrum, i = intruder, d = descriptor, b = boltzmann weighting
def evaluate_loss_sidb(
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
        w_0, w_1
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
        Which descriptors to use for the descriptor distance calculation.
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

    if fcivec is None and "fcivec" in cache:
        fcivec = cache["fcivec"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )
    cache["fcivec"] = fcivec

    dist_des = loss_function.descriptor_distance(
        ai_df, descriptors, matches=matches, norm=norm
    )
    dist_energy = loss_function.descriptor_distance(
        ai_df, descriptors, matches=["energy"], norm=norm
    )

    distance = w_0 * dist_energy + w_1 * dist_des

    penalty = np.maximum(0, max_ai_energy - descriptors['energy'])

    npenalty = nroots - len(ai_df)
    distance = np.vstack((distance, np.tile(w_0 * penalty, (npenalty, 1))))  # replace 6 with nroots-len(ai_df)

    boltzmann_weights = np.concatenate((boltzmann_weights, np.ones((npenalty))))

    row_ind, col_ind = linear_sum_assignment(distance)

    loss = np.sum(boltzmann_weights * distance[row_ind, col_ind])

    return {
        "loss": loss,
        "sloss": np.sum(dist_energy[row_ind[:len(ai_df)], col_ind[:len(ai_df)]]),
        "dloss": np.sum(dist_des[row_ind[:len(ai_df)], col_ind[:len(ai_df)]]),
        "penalty": np.sum(np.tile(penalty, (npenalty, 1))[row_ind[len(ai_df):] - len(ai_df), col_ind[len(ai_df):]]), # NEEDS TO BE FIXED
        "descriptors": descriptors,
        "distance": distance,
        "row_ind": row_ind,
        "col_ind": col_ind,
        "params": params,
    }


def evaluate_loss_sid(
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
    w_0 = weights[0]
    w_1 = weights[1]

    if fcivec is None and "fcivec" in cache:
        fcivec = cache["fcivec"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )
    cache["fcivec"] = fcivec

    dist_des = loss_function.descriptor_distance(
        ai_df, descriptors, matches=matches, norm=norm
    )
    dist_energy = loss_function.descriptor_distance(
        ai_df, descriptors, matches=["energy"], norm=norm
    )

    distance = w_0 * dist_energy + w_1 * dist_des

    penalty = np.maximum(0, max_ai_energy - descriptors['energy'])

    npenalty = nroots - len(ai_df)
    distance = np.vstack((distance, np.tile(w_0 * penalty, (npenalty, 1))))  # replace 6 with nroots-len(ai_df)

    boltzmann_weights = np.concatenate((boltzmann_weights, np.ones((npenalty))))

    row_ind, col_ind = linear_sum_assignment(distance)

    loss = np.sum(distance[row_ind, col_ind])

    return {
        "loss": loss,
    }


def evaluate_loss_si(
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
    w_0 = weights[0]
    w_1 = weights[1]

    if fcivec is None and "fcivec" in cache:
        fcivec = cache["fcivec"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )
    cache["fcivec"] = fcivec

    dist_des = loss_function.descriptor_distance(
        ai_df, descriptors, matches=matches, norm=norm
    )
    dist_energy = loss_function.descriptor_distance(
        ai_df, descriptors, matches=["energy"], norm=norm
    )

    distance = dist_energy

    penalty = np.maximum(0, max_ai_energy - descriptors['energy'])

    npenalty = nroots - len(ai_df)
    distance = np.vstack((distance, np.tile(penalty, (npenalty, 1))))  # replace 6 with nroots-len(ai_df)

    boltzmann_weights = np.concatenate((boltzmann_weights, np.ones((npenalty))))

    row_ind, col_ind = linear_sum_assignment(distance)

    loss = np.sum(distance[row_ind, col_ind])

    return {
        "loss": loss,
    }


def evaluate_loss_sd(
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
    w_0 = weights[0]
    w_1 = weights[1]

    if fcivec is None and "fcivec" in cache:
        fcivec = cache["fcivec"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )
    cache["fcivec"] = fcivec

    dist_des = loss_function.descriptor_distance(
        ai_df, descriptors, matches=matches, norm=norm
    )
    dist_energy = loss_function.descriptor_distance(
        ai_df, descriptors, matches=["energy"], norm=norm
    )

    distance = w_0 * dist_energy + w_1 * dist_des

    penalty = np.maximum(0, max_ai_energy - descriptors['energy'])

    npenalty = nroots - len(ai_df)
    #distance = np.vstack((distance, np.tile(w_0 * penalty, (npenalty, 1))))  # replace 6 with nroots-len(ai_df)

    boltzmann_weights = np.concatenate((boltzmann_weights, np.ones((npenalty))))

    row_ind, col_ind = linear_sum_assignment(distance)

    loss = np.sum(distance[row_ind, col_ind])

    return {
        "loss": loss,
    }


def evaluate_loss_s(
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
    w_0 = weights[0]
    w_1 = weights[1]

    if fcivec is None and "fcivec" in cache:
        fcivec = cache["fcivec"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )
    cache["fcivec"] = fcivec

    dist_des = loss_function.descriptor_distance(
        ai_df, descriptors, matches=matches, norm=norm
    )
    dist_energy = loss_function.descriptor_distance(
        ai_df, descriptors, matches=["energy"], norm=norm
    )

    distance = dist_energy

    penalty = np.maximum(0, max_ai_energy - descriptors['energy'])

    npenalty = nroots - len(ai_df)
    #distance = np.vstack((distance, np.tile(w_0 * penalty, (npenalty, 1))))  # replace 6 with nroots-len(ai_df)

    boltzmann_weights = np.concatenate((boltzmann_weights, np.ones((npenalty))))

    row_ind, col_ind = linear_sum_assignment(distance)

    loss = np.sum(distance[row_ind, col_ind])

    return {
        "loss": loss,
    }


if __name__ == "__main__":
    ngrid = 50

    eps = -0.498
    #generate_t_U(0.0, -0.05, 0.1, 0.4, eps=eps, ngrid=ngrid, outfile=f"loss_grid{ngrid}_eps{eps}.csv")
    generate_t_U(0.05, -0.05, 0.2, 0.4, eps=eps, ngrid=ngrid, outfile=f"loss_grid{ngrid}_eps{eps}_big_w0-0.7.csv")
