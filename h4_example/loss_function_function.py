import mlflow
import pandas as pd
import statsmodels.api as sm
import h5py
import numpy as np
from rich import print
from scipy.optimize import linear_sum_assignment, minimize, curve_fit
import solver

cache = {}  # For speedup during mulitiple ED during model optimization

# Parameter functions
def exponential(rs, d, r_0): # 2 parameters
    x = np.exp(-d * (rs - r_0)) - 0.5  # exponential
    return x

def sigmoid(rs, C, d, r_0): # 3 parameters
    x = (C / (1 + np.exp(-d * (rs - r_0)))) - C 
    return x

def ploynomial3(rs, a, b, c, d):  # 4 parameters
    x = a + b*rs + c*rs**2 + d*rs**3  # 3 degree polynomial
    return x

def ploynomial4(rs, a, b, c, d, e):  # 5 parameters
    x = a + b*rs + c*rs**2 + d*rs**3 + e*rs**4  # 4 degree polynomial
    return x

def func_E0(rs, d, r_0):  # 2 parameters
    x = np.exp(-d * (rs - r_0)) - 0.5  # exponential
    return x

def func_t(rs, C, d, r_0):  # 3 parameters
    # C is carrying capacity, i.e. min(t)
    x = (C / (1 + np.exp(-d * (rs - r_0)))) - C  # sigmoid, logistic
    return x

def func_U(rs, a, r_0, c):  # 2 parameters
    x = -a*1*(rs - r_0)**-1 + c  # 1/r function
    return x


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
    onebody,
    twobody,
    ai_df,
    max_ai_energy,
    nroots,
    matches,
    fcivec,
    norm,
    r,
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
        fcivec = cache[f"fcivec_{r}"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=fcivec
    )
    cache[f"fcivec_{r}"] = fcivec

    dist_des = descriptor_distance(
        ai_df, descriptors, matches=matches, norm=norm
    )
    dist_energy = descriptor_distance(
        ai_df, descriptors, matches=["energy"], norm=norm
    )

    distance = w_0 * dist_energy + w_1 * dist_des

    penalty = np.maximum(0, max_ai_energy - descriptors['energy'])

    npenalty = nroots - len(ai_df)
    distance = np.vstack((distance, np.tile(w_0 * penalty, (npenalty, 1))))

    row_ind, col_ind = linear_sum_assignment(distance)

    loss = np.sum(distance[row_ind, col_ind])

    dloss_row_inds = row_ind[np.where(row_ind < len(ai_df))[0]]
    dloss_col_inds = col_ind[np.where(row_ind < len(ai_df))[0]]

    penalty_row_inds = row_ind[np.where(row_ind >= len(ai_df))[0]]
    penalty_col_inds = col_ind[np.where(row_ind >= len(ai_df))[0]]

    energy_loss = np.sum(dist_energy[dloss_row_inds, dloss_col_inds])
    spectrum_RMSE = np.sqrt(np.mean((ai_df['energy'].values[dloss_row_inds] - descriptors['energy'][dloss_col_inds])**2))

    # Debugging information to test loss
    #print()
    #print("loss ", loss)
    #print("params", params.values)

    return {
        "norm": norm,
        'energy_loss': energy_loss,
        "loss": loss,
        "sloss": np.mean(dist_energy[dloss_row_inds, dloss_col_inds])*norm['energy'],
        "dloss": np.mean(dist_des[dloss_row_inds, dloss_col_inds]),
        "penalty": np.sum(np.tile(penalty, (npenalty, 1))[:, penalty_col_inds]),
        "descriptors": descriptors,
        "distance": distance,
        "row_ind": row_ind,
        "col_ind": col_ind,
        "params": params,
        'Spectrum RMSE (Ha)': spectrum_RMSE,
    }

def evaluate_loss_para_function(
    x0,
    x0_ind,
    rs,
    keys,
    weights,
    onebody,
    twobody,
    ai_df_rs,
    max_ai_energy_rs,
    nroots,
    matches,
    fcivec,
    norm_rs,
):

    print(keys) # E0, t, U

    losses = {}
    sum_loss = 0
    sum_spec_rmse = 0

    for r in rs:

        #print(r)

        E0 = func_E0(r, x0[x0_ind[0]][0], x0[x0_ind[0]][1])
        t = func_t(r, x0[x0_ind[1]][0], x0[x0_ind[1]][1], x0[x0_ind[1]][2])
        U = func_U(r, x0[x0_ind[2]][0], x0[x0_ind[2]][1], x0[x0_ind[2]][2])

        params = [E0, t, U]

        losses[f'r{r}'] = evaluate_loss(params,
                                           keys,
                                           weights,
                                           onebody,
                                           twobody,
                                           ai_df_rs[f'r{r}'],
                                           max_ai_energy_rs[f'r{r}'],
                                           nroots,
                                           matches,
                                           None,
                                           norm_rs[f'r{r}'],
                                           r)
        sum_loss += losses[f'r{r}']["loss"]
        sum_spec_rmse += losses[f'r{r}']["Spectrum RMSE (Ha)"]

    losses["sum_loss"] = sum_loss
    mean_over_r_spectrum_rmse_ha = np.mean(sum_spec_rmse)
    losses["Mean over r Spectrum RMSE (Ha)"] = mean_over_r_spectrum_rmse_ha

    print(x0)
    print(sum_loss)

    return losses

# maps for train states first, then maps the rest of the model states to test states.
def CV_evaluate_loss(
    params: np.ndarray,
    keys,
    weights,
    onebody,
    twobody,
    ai_df,
    max_ai_energy,
    nroots,
    matches,
    fcivec,
    norm,
    train_states,
    test_states,
    r,
) -> float:

    w_0 = weights[0]
    w_1 = weights[1]

    if fcivec is None and "fcivec" in cache:
        fcivec = cache[f"fcivec_{r}"]

    params = pd.Series(params, index=keys)

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, ci0=None
    )
    cache[f"fcivec_{r}"] = fcivec

    dist_des = descriptor_distance(ai_df, descriptors, matches=matches, norm=norm)
    dist_energy = descriptor_distance(ai_df, descriptors, matches=["energy"], norm=norm)

    distance = w_0 * dist_energy + w_1 * dist_des

    penalty = np.maximum(0, max_ai_energy - descriptors['energy'])

    npenalty = nroots - len(ai_df)
    distance_train = np.vstack((distance[train_states], np.tile(w_0 * penalty, (npenalty, 1))))
    distance = np.vstack((distance, np.tile(w_0 * penalty, (npenalty, 1))))

    row_ind, col_ind = linear_sum_assignment(distance_train)
    #print("loss before reindexing", np.sum(distance_train[row_ind, col_ind]))
    for p in np.sort(test_states):  # Reindexing row so that it is used on distance matrix
        ind = np.where(row_ind == p)[0][0]
        for i in range(ind, len(row_ind)):
            row_ind[i] += 1

    loss = np.sum(distance[row_ind, col_ind])
    #print("loss after reindexing", loss)

    # mapping the left over train states with left over model states

    x = np.delete(np.arange(0, nroots), col_ind)
    dist = distance[test_states][:, x]
    rows, cols = linear_sum_assignment(dist)

    test_loss = np.sum(dist[rows, cols])

    train_row_inds = row_ind[np.where(row_ind < len(ai_df))[0]]
    train_col_inds = col_ind[np.where(row_ind < len(ai_df))[0]]
    test_row_inds = test_states[rows]
    test_col_inds = x[cols]
    penalty_row_inds = row_ind[np.where(row_ind >= len(ai_df))[0]]
    penalty_col_inds = col_ind[np.where(row_ind >= len(ai_df))[0]]

    ntest = len(train_states)
    ntrain = len(test_states)
    row_ind, col_ind = linear_sum_assignment(distance)
    row_ind = row_ind[:ntest + ntrain]
    col_ind = col_ind[:ntest + ntrain]
    better_test_loss = np.sum(dist_energy[test_row_inds, test_col_inds])
    # np.sqrt(np.mean((ai_df['energy'].values[test_states] - dist_energy[col_ind[test_states]])**2))
    energy_loss = np.sum(dist_energy[row_ind, col_ind])
    # np.sqrt(np.mean( (ai_df['energy'].values[row_ind] - dist_energy[col_ind])**2))

    spectrum_RMSE_train = np.sqrt(np.mean((ai_df['energy'].values[train_row_inds] - descriptors['energy'][train_col_inds])**2))
    spectrum_RMSE_val = np.sqrt(np.mean((ai_df['energy'].values[test_row_inds] - descriptors['energy'][test_col_inds])**2))
    spectrum_RMSE_trainval = np.sqrt(np.mean((ai_df['energy'].values[row_ind] - descriptors['energy'][col_ind])**2))

    return {
        "train_loss": loss / 2,  # train loss
        "test_loss": test_loss / 3,
        "norm": norm,
        "train_sloss": np.sum(dist_energy[train_row_inds, train_col_inds])*norm['energy']/ntrain,
        "train_dloss": np.sum(dist_des[train_row_inds, train_col_inds]),
        "test_sloss": np.sum(dist_energy[test_row_inds, test_col_inds]),
        "test_dloss": np.sum(dist_des[test_row_inds, test_col_inds]),
        "penalty": np.sum(np.tile(penalty, (npenalty, 1))[:, penalty_col_inds]),
        "descriptors": descriptors,
        "distance": distance,
        "row_ind": row_ind,
        "col_ind": col_ind,
        "params": params,
        'test_rows': np.array(test_states)[rows],
        'test_cols': x[cols],
        'Energy test rms error (eV)':27.2114*better_test_loss,
        'Test loss': better_test_loss,
        'energy_loss': energy_loss,
        'Spectrum RMSE Train (Ha)': spectrum_RMSE_train,
        'Spectrum RMSE Val States (Ha)': spectrum_RMSE_val,
        'Spectrum RMSE Val (Ha)': spectrum_RMSE_trainval,}


def optimize_function(*args, **kwargs):
    return evaluate_loss(*args, **kwargs)["loss"]


def optimize_CV_function(*args, **kwargs):
    return CV_evaluate_loss(*args, **kwargs)["train_loss"]


def evaluate_loss_CV_para_function(
    x0,
    x0_ind,
    rs,
    keys,
    weights,
    onebody,
    twobody,
    ai_df_rs,
    max_ai_energy_rs,
    nroots,
    matches,
    fcivec,
    norm_rs,
    train_states_rs,
    test_states_rs,
):

    print(keys) # E0, t, U

    losses = {}
    sum_loss = 0
    sum_spec_rmse_train = 0
    sum_spec_rmse_val = 0

    for r in rs:

        #print(r)

        E0 = func_E0(r, x0[x0_ind[0]][0], x0[x0_ind[0]][1])
        t = func_t(r, x0[x0_ind[1]][0], x0[x0_ind[1]][1], x0[x0_ind[1]][2])
        U = func_U(r, x0[x0_ind[2]][0], x0[x0_ind[2]][1], x0[x0_ind[2]][2])

        params = [E0, t, U]

        losses[f'r{r}'] = CV_evaluate_loss(params,
                                           keys,
                                           weights,
                                           onebody,
                                           twobody,
                                           ai_df_rs[f'r{r}'],
                                           max_ai_energy_rs[f'r{r}'],
                                           nroots,
                                           matches,
                                           None,
                                           norm_rs[f'r{r}'],
                                           train_states_rs[f'r{r}'],
                                           test_states_rs[f'r{r}'],
                                           r)
        sum_loss += losses[f'r{r}']["train_loss"]
        sum_spec_rmse_train += losses[f'r{r}']["Spectrum RMSE Train (Ha)"]
        sum_spec_rmse_val += losses[f'r{r}']["Spectrum RMSE Val (Ha)"]

    losses["sum_loss"] = sum_loss
    mean_over_r_spectrum_rmse_ha_train = np.mean(sum_spec_rmse_train)
    mean_over_r_spectrum_rmse_ha_val = np.mean(sum_spec_rmse_val)
    losses["Mean over r Spectrum RMSE (Ha) - Train"] = mean_over_r_spectrum_rmse_ha_train
    losses["Mean over r Spectrum RMSE (Ha) - Validation"] = mean_over_r_spectrum_rmse_ha_val

    print(x0)
    print(sum_loss)

    return losses


def optimize_CV_para_function(*args, **kwargs):
    return evaluate_loss_CV_para_function(*args, **kwargs)["sum_loss"]


def mapping(
    onebody: dict,
    twobody: dict,
    onebody_params: list,
    twobody_params: list,
    ai_df_rs: dict[str, pd.DataFrame],
    nroots: int,
    outfile: str,
    matches: list,
    train_rs: list,
    test_rs: list,
    weights: list,
    beta: float,
    p: int,
    guess_params=None,
    clip_val=1,
    niter_opt=1000,
    tol_opt=1e-7,
    maxfev_opt=10000, 
):  
    ai_df_train_rs = {}
    ai_df_test_rs = {}
    max_ai_energy_rs = {}

    train_states_rs = {}
    test_states_rs = {}

    for r in (train_rs + test_rs):
        ai_df = ai_df_rs[f'r{r}']
        p_out_states = np.random.choice(np.arange(0, len(ai_df)), size=p, replace=False)
        #  p_out_states = np.array([p]) # this p is for 30-k_groups, leaving out data 1-by-1
        ai_df_train_rs[f'r{r}'] = ai_df.drop(p_out_states, axis=0)
        ai_df_test_rs[f'r{r}'] = ai_df.loc[p_out_states]
        max_ai_energy_rs[f'r{r}'] = np.max(ai_df["energy"])

        train_states_rs[f'r{r}'] = np.delete(np.arange(0, len(ai_df)), p_out_states)
        test_states_rs[f'r{r}'] = p_out_states
    
    #boltzmann_weights_train = give_boltzmann_weights(ai_df_train["energy"], ai_df["energy"][0], beta)
    #print(f"Boltzmann weights for beta {beta}: {boltzmann_weights_train}")
    #boltzmann_weights_test = give_boltzmann_weights(ai_df_test["energy"], ai_df["energy"][0], beta)
    #print(f"Boltzmann weights for beta {beta}: {boltzmann_weights_test}")

    # Starting parameter guess

    #params_rs = []
    # Can only use Hubbard model at the moment, TODO: generalize to parameters
    E0_rs = []
    t_rs = []
    U_rs = []

    for r in (train_rs):
        ai_df = ai_df_rs[f'r{r}']
        dmd = sm.OLS(ai_df["energy"], ai_df[onebody_params + twobody_params]).fit()
        #print(dmd.summary())
        params = dmd.params.copy()
        params['E0'] = ( ai_df["energy"][0] - (params['t']*ai_df["t"][0] + params['U']*ai_df["U"][0]) )/4 
        #print("New E0: ", params['E0'])
        E0_rs.append(params['E0'])
        t_rs.append(params['t'])
        U_rs.append(params['U'])

    print("DMD parameters (E0, t, U):")
    print("E0 ", E0_rs)
    print("t ", t_rs)
    print("U", U_rs)

    print("Parameter function initial variables (E0, t, U):")
    # Set up guess functions -------
    popt_E0, pcov_E0 = curve_fit(func_E0, train_rs, E0_rs)
    popt_t, pcov_t = curve_fit(func_t, train_rs, t_rs)
    popt_U, pcov_U = curve_fit(func_U, train_rs, U_rs) # Probably rewrite using dictionaries

    print("E0 : d, r_0 ", popt_E0)
    print("t  : C, d, r_0 ", popt_t)
    print("U : a, r_0, c", popt_U) 
    x0_ind = [[0, 1], [2, 3, 4], [5, 6, 7]]
    x0 = np.concatenate((popt_E0, popt_t, popt_U))

    norm_rs = {}

    for r in (train_rs + test_rs):
        ai_df = ai_df_rs[f'r{r}']
        ai_var = np.var(ai_df, axis=0)
        #print("Before clipping:\n", ai_var)
        des_var = ai_var.loc[ai_var.index != 'energy']
        des_var = np.clip(des_var, clip_val, np.max(des_var))  # clip so that small variances are set to 1
        ai_var.loc[ai_var.index != 'energy'] = des_var
        #print("After clipping:\n", ai_var)
        norm_rs[f'r{r}'] = 2 * ai_var

    keys = params.keys()

    # OPTMIZATION LOOP START
    print("Starting optimization")

    xmin = minimize(
        optimize_CV_para_function,
        x0,
        args=(
            x0_ind,
            train_rs,
            keys,
            weights,
            onebody,
            twobody,
            ai_df_rs,
            max_ai_energy_rs,
            nroots,
            matches,
            None,
            norm_rs,
            train_states_rs,
            test_states_rs,
        ),
        jac="3-point",
        method="Powell",
        tol=tol_opt,
        options={"maxiter": niter_opt, "maxfev": maxfev_opt},
    )

    print(xmin.nit, xmin.nfev)
    print(xmin.message)
    print("function value", xmin.fun)
    print("parameters", xmin.x)

    print("Evaluate train data after optimization:")
    data = evaluate_loss_CV_para_function(xmin.x,
                                     x0_ind,
                                     train_rs,
                                     keys,
                                     weights,
                                     onebody,
                                     twobody,
                                     ai_df_rs,
                                     max_ai_energy_rs,
                                     nroots,
                                     matches,
                                     None,
                                     norm_rs,
                                     train_states_rs,
                                     test_states_rs)

    E0_test_rs = []
    t_test_rs = []
    U_test_rs = []
    for r in (test_rs):
        ai_df = ai_df_rs[f'r{r}']
        dmd = sm.OLS(ai_df["energy"], ai_df[onebody_params + twobody_params]).fit()
        #print(dmd.summary())
        params = dmd.params.copy()
        params['E0'] = ( ai_df["energy"][0] - (params['t']*ai_df["t"][0] + params['U']*ai_df["U"][0]) )/4 
        #print("New E0: ", params['E0'])
        E0_test_rs.append(params['E0'])
        t_test_rs.append(params['t'])
        U_test_rs.append(params['U'])

    data_test = evaluate_loss_para_function(xmin.x,
                                     x0_ind,
                                     test_rs,
                                     keys,
                                     weights,
                                     onebody,
                                     twobody,
                                     ai_df_rs,
                                     max_ai_energy_rs,
                                     nroots,
                                     matches,
                                     None,
                                     norm_rs)

    with h5py.File(outfile, "w") as f:
        f["train_rs"] = train_rs
        f["test_rs"] = test_rs
        f["para_w_0"] = weights[0]
        f["para_w_1"] = weights[1]
        f["loss"] = xmin.fun
        f["function parameters, E0, t, U"] = xmin.x
        f["Mean over r Spectrum RMSE (Ha) - Train"] = data["Mean over r Spectrum RMSE (Ha) - Train"]
        f["Mean over r Spectrum RMSE (Ha) - Validation"] = data["Mean over r Spectrum RMSE (Ha) - Validation"]
        f["Mean over r Spectrum RMSE (Ha) - Test"] = data_test["Mean over r Spectrum RMSE (Ha)"]

        mlflow.log_metrics({
            "loss": xmin.fun,
            "mean_over_r_spectrum_rmse_ha_train": data["Mean over r Spectrum RMSE (Ha) - Train"],
            "mean_over_r_spectrum_rmse_ha_val": data["Mean over r Spectrum RMSE (Ha) - Validation"],
            "mean_over_r_spectrum_rmse_ha_test": data_test["Mean over r Spectrum RMSE (Ha)"],
            "sum-loss": data["sum_loss"],
            "para_w_0": weights[0],
            "para_w_1": weights[1],
        })


        for i, r in enumerate(train_rs):
            data_r = data[f"r{r}"]

            for k in onebody_params + twobody_params:
                if k == 'E0':
                    f[f"r{r}/" + "dmd_params/" + k] = E0_rs[i]
                if k == 't':
                    f[f"r{r}/" + "dmd_params/" + k] = t_rs[i]
                if k == 'U':
                    f[f"r{r}/" + "dmd_params/" + k] = U_rs[i]

            ai_df = ai_df_rs[f"r{r}"]
            f[f"r{r}/" + "ai_spectrum_range (Ha)"] = np.max(ai_df["energy"]) - np.min(ai_df["energy"])

            f[f"r{r}/" + "nstates_train"] = len(train_states_rs[f"r{r}"])
            f[f"r{r}/" + "nstates_test"] = len(test_states_rs[f"r{r}"])
            f[f"r{r}/" + "state_ind_for_test"] = test_states_rs[f"r{r}"]

            for k in data_r:
                if k == "descriptors":
                    for kk in data_r[k]:
                        f[f"r{r}/" + "train/" + k + "/" + kk] = data_r[k][kk]
                elif k == "params":
                    for kk in onebody_params + twobody_params:
                        f[f"r{r}/" + "rdmd_params/" + kk] = data_r[k][kk]
                else:
                    f[f"r{r}/" + k] = data_r[k]

        for i, r in enumerate(test_rs):
            data_r = data_test[f"r{r}"]

            for k in onebody_params + twobody_params:
                if k == 'E0':
                    f[f"r{r}/" + "dmd_params/" + k] = E0_test_rs[i]
                if k == 't':
                    f[f"r{r}/" + "dmd_params/" + k] = t_test_rs[i]
                if k == 'U':
                    f[f"r{r}/" + "dmd_params/" + k] = U_test_rs[i]

            ai_df = ai_df_rs[f"r{r}"]
            f[f"r{r}/" + "ai_spectrum_range (Ha)"] = np.max(ai_df["energy"]) - np.min(ai_df["energy"])

            for k in data_r:
                if k == "descriptors":
                    for kk in data_r[k]:
                        f[f"r{r}/" + "test/" + k + "/" + kk] = data_r[k][kk]
                elif k == "params":
                    for kk in onebody_params + twobody_params:
                        f[f"r{r}/" + "rdmd_params/" + kk] = data_r[k][kk]
                else:
                    f[f"r{r}/" + k] = data_r[k]

        f["iterations"] = xmin.nit
        f["termination_message"] = xmin.message

    mlflow.log_artifact(outfile)