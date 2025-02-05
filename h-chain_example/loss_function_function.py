import pandas as pd
import statsmodels.api as sm
import h5py
import numpy as np
from rich import print
from scipy.optimize import linear_sum_assignment, minimize, curve_fit
import solver
from inspect import signature

cache = {}  # For speedup during mulitiple ED during model optimization


# Parameter functions
def constant(rs, c):
    x = c
    return x

def exponential(rs, a, d, r_0, c): # 4 parameters
    x = a*np.exp(-d*(rs - r_0)) + c  # exponential
    return x

def sigmoid(rs, C, d, r_0): # 3 parameters
    x = (C / (1 + np.exp(-d * (rs - r_0)))) - C 
    return x

def polynomial3(rs, a, b, c, d):  # 4 parameters
    x = a + b*rs + c*rs**2 + d*rs**3  # 3 degree polynomial
    return x

def polynomial4(rs, a, b, c, d, e):  # 5 parameters
    x = a + b*rs + c*rs**2 + d*rs**3 + e*rs**4  # 4 degree polynomial
    return x

def polynomial5(rs, a, b, c, d, e, f):  # 6 parameters
    x = a + b*rs + c*rs**2 + d*rs**3 + e*rs**4 + f*rs**5 # 5 degree polynomial
    return x

def func_E0(rs, d, r_0):  # 2 parameters
    x = np.exp(-d * (rs - r_0)) - (0.5*27.2114)  # exponential
    return x

def func_t(rs, C, d, r_0):  # 3 parameters
    # C is carrying capacity, i.e. min(t)
    x = (C / (1 + np.exp(-d * (rs - r_0)))) - C  # sigmoid, logistic
    return x

def func_U(rs, a, r_0, c):  # 3 parameters
    x = a*(rs - r_0)**-1 + c  # 1/r function
    return x

def func_U_linear(rs, a, r_0, c, b, r_1):  # 5  parameters
    x = a*(rs - r_0)**-1 + c + b*(rs - r_1) # 1/r function + linear function
    return x

def func_U_linear_fixed(rs, a, r_0, b, r_1):  # 4  parameters
    x = a*(rs - r_0)**-1 + 0.50 + b*(rs - r_1)**-2 # 1/r function + linear function
    return x

def unpack_constant(rs, i, params):
    c = params[0]
    return constant(rs, c)

def unpack_exponential(rs, i, params):
    d = params[0]
    r_0 = params[1]
    c = params[2]
    return  exponential(rs, d, r_0, c)

def unpack_sigmoid(rs, i, params):
    C = params[0]
    d = params[1]
    r_0 = params[2]
    return sigmoid(rs, C, d, r_0)

def unpack_polynomial3(rs, i, params):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    return polynomial3(rs, a, b, c, d)

def unpack_polynomial4(rs, i, params):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    e = params[4]
    return polynomial4(rs, a, b, c, d, e)

def unpack_polynomial5(rs, i, params):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    e = params[4]
    f = params[5]
    return polynomial5(rs, a, b, c, d, e, f)

def unpack_func_E0(rs, i, params):
    d = params[0]
    r_0 = params[1]
    return func_E0(rs, d, r_0)

def unpack_func_t(rs, i, params):
    C = params[0]
    d = params[1]
    r_0 = params[2]
    return func_t(rs, C, d, r_0)

def unpack_func_U(rs, i, params):
    a = params[0]
    r_0 = params[1]
    c = params[2]
    return func_U(rs, a, r_0, c)

def unpack_func_U_linear(rs, i, params):
    a = params[0]
    r_0 = params[1]
    c = params[2]
    b = params[3]
    r_1 = params[4]
    return func_U_linear(rs, a, r_0, c, b, r_1)

def unpack_func_U_linear_fixed(rs, i, params):
    a = params[0]
    r_0 = params[1]
    b = params[2]
    r_1 = params[3]
    return func_U_linear_fixed(rs, a, r_0, b, r_1)

def unpack_independent(rs, i, params):
    return params[i]

function_dict = {
    'constant' : constant,
    'func_E0' : func_E0,
    'func_t' : func_t,
    'func_U' : func_U,
    'func_U_linear' : func_U_linear,
    'func_U_linear_fixed' : func_U_linear_fixed,
    'exponential' : exponential,
    'sigmoid' : sigmoid,
    'polynomial3' : polynomial3,
    'polynomial4' : polynomial4,
    'polynomial5' : polynomial5,
}

unpack_func_dict = {
    'constant' : unpack_constant,
    'func_E0' : unpack_func_E0,
    'func_t' : unpack_func_t,
    'func_U' : unpack_func_U,
    'func_U_linear' : unpack_func_U_linear,
    'func_U_linear_fixed' : unpack_func_U_linear_fixed,
    'exponential' : unpack_exponential,
    'sigmoid' : unpack_sigmoid,
    'polynomial3' : unpack_polynomial3,
    'polynomial4' : unpack_polynomial4,
    'polynomial5' : unpack_polynomial5,
    'independent' : unpack_independent,
}

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

    N = onebody[keys[0]].shape[0] # The first key is a onebody key

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, norb=N, nelec=(N//2,N//2), ci0=fcivec
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
        'Spectrum RMSE': spectrum_RMSE,
    }

def evaluate_loss_para_function(
    x0,
    x0_ind,
    param_functions,
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

    #print(keys) # E0, t, U

    losses = {}
    sum_loss = 0
    sum_spec_rmse = 0

    for i, r in enumerate(rs):

        #print(r)

        params = []
        for j, param in enumerate(keys):
            params.append(unpack_func_dict[param_functions[j]](r, i, x0[x0_ind[j] : x0_ind[j+1]]))

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
        sum_spec_rmse += losses[f'r{r}']["Spectrum RMSE"]

    losses["sum_loss"] = sum_loss
    mean_over_r_spectrum_rmse_ha = np.mean(sum_spec_rmse)
    losses["Mean over r Spectrum RMSE"] = mean_over_r_spectrum_rmse_ha

    #print(x0)
    #print(sum_loss)

    return losses

# maps for train states first, then maps the rest of the model states to validation states.
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
    val_states,
    r,
) -> float:

    w_0 = weights[0]
    w_1 = weights[1]

    if fcivec is None and "fcivec" in cache:
        fcivec = cache[f"fcivec_{r}"]

    params = pd.Series(params, index=keys)

    N = onebody[keys[0]].shape[0]

    descriptors, fcivec = solver.solve_effective_hamiltonian(
        onebody, twobody, params, nroots=nroots, norb=N, nelec=(N//2,N//2), ci0=None
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
    for p in np.sort(val_states):  # Reindexing row so that it is used on distance matrix
        ind = np.where(row_ind == p)[0][0]
        for i in range(ind, len(row_ind)):
            row_ind[i] += 1

    loss = np.sum(distance[row_ind, col_ind])
    #print("loss after reindexing", loss)

    # mapping the left over train states with left over model states

    x = np.delete(np.arange(0, nroots), col_ind)
    dist = distance[val_states][:, x]
    rows, cols = linear_sum_assignment(dist)

    val_loss = np.sum(dist[rows, cols])

    train_row_inds = row_ind[np.where(row_ind < len(ai_df))[0]]
    train_col_inds = col_ind[np.where(row_ind < len(ai_df))[0]]
    val_row_inds = val_states[rows]
    val_col_inds = x[cols]
    penalty_row_inds = row_ind[np.where(row_ind >= len(ai_df))[0]]
    penalty_col_inds = col_ind[np.where(row_ind >= len(ai_df))[0]]

    ntrain = len(train_states)
    nval = len(val_states)
    row_ind, col_ind = linear_sum_assignment(distance)
    row_ind = row_ind[:nval + ntrain]
    col_ind = col_ind[:nval + ntrain]
    better_val_loss = np.sum(dist_energy[val_row_inds, val_col_inds])
    # np.sqrt(np.mean((ai_df['energy'].values[test_states] - dist_energy[col_ind[test_states]])**2))
    energy_loss = np.sum(dist_energy[row_ind, col_ind])
    # np.sqrt(np.mean( (ai_df['energy'].values[row_ind] - dist_energy[col_ind])**2))

    spectrum_RMSE_train = np.sqrt(np.mean((ai_df['energy'].values[train_row_inds] - descriptors['energy'][train_col_inds])**2))
    spectrum_RMSE_val = np.sqrt(np.mean((ai_df['energy'].values[val_row_inds] - descriptors['energy'][val_col_inds])**2))
    spectrum_RMSE_trainval = np.sqrt(np.mean((ai_df['energy'].values[row_ind] - descriptors['energy'][col_ind])**2))

    return {
        "train_loss": loss / 2,  # train loss
        "val_loss": val_loss / 3,
        "norm": norm,
        "train_sloss": np.sum(dist_energy[train_row_inds, train_col_inds])*norm['energy']/ntrain,
        "train_dloss": np.sum(dist_des[train_row_inds, train_col_inds]),
        "val_sloss": np.sum(dist_energy[val_row_inds, val_col_inds]),
        "val_dloss": np.sum(dist_des[val_row_inds, val_col_inds]),
        "penalty": np.sum(np.tile(penalty, (npenalty, 1))[:, penalty_col_inds]),
        "descriptors": descriptors,
        "distance": distance,
        "row_ind": row_ind,
        "col_ind": col_ind,
        "params": params,
        'val_rows': np.array(val_states)[rows],
        'val_cols': x[cols],
        'Energy val rms error (eV)':27.2114*better_val_loss,
        'val loss': better_val_loss,
        'energy_loss': energy_loss,
        'Spectrum RMSE Train': spectrum_RMSE_train,
        'Spectrum RMSE Val States': spectrum_RMSE_val,
        'Spectrum RMSE Val': spectrum_RMSE_trainval,}


def optimize_function(*args, **kwargs):
    return evaluate_loss(*args, **kwargs)["loss"]


def optimize_CV_function(*args, **kwargs):
    return CV_evaluate_loss(*args, **kwargs)["train_loss"]


def evaluate_loss_CV_para_function(
    x0,
    x0_ind,
    param_functions,
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
    val_states_rs,
):

    #print(keys) # E0, t, U

    losses = {}
    sum_loss = 0
    sum_spec_rmse_train = 0
    sum_spec_rmse_val = 0

    for i,r in enumerate(rs):

        #print(r)
        params = []
        for j, param in enumerate(keys):
            params.append(unpack_func_dict[param_functions[j]](r, i, x0[x0_ind[j] : x0_ind[j+1]]))

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
                                           val_states_rs[f'r{r}'],
                                           r)
        sum_loss += losses[f'r{r}']["train_loss"]
        sum_spec_rmse_train += losses[f'r{r}']["Spectrum RMSE Train"]
        sum_spec_rmse_val += losses[f'r{r}']["Spectrum RMSE Val"]

    losses["sum_loss"] = sum_loss
    mean_over_r_spectrum_rmse_ha_train = np.mean(sum_spec_rmse_train)
    mean_over_r_spectrum_rmse_ha_val = np.mean(sum_spec_rmse_val)
    losses["Mean over r Spectrum RMSE - Train"] = mean_over_r_spectrum_rmse_ha_train
    losses["Mean over r Spectrum RMSE - Validation"] = mean_over_r_spectrum_rmse_ha_val

    #print(x0)
    #print(sum_loss)

    return losses


def optimize_CV_para_function(*args, **kwargs):
    return evaluate_loss_CV_para_function(*args, **kwargs)["sum_loss"]


def setup_train(
    onebody: dict,
    twobody: dict,
    onebody_params: list,
    twobody_params: list,
    ai_df_rs: dict[str, pd.DataFrame],
    nroots: int,
    outfile: str,
    matches: list,
    train_rs: list,
    param_functions: list,
    weights: list,
    p: int,
    guess_params=None,
    clip_val=1,
    niter_opt=1000,
    tol_opt=1e-7,
    maxfev_opt=10000, 
):  
    ai_df_train_rs = {}
    max_ai_energy_rs = {}

    train_states_rs = {}
    val_states_rs = {}

    for r in (train_rs):
        ai_df = ai_df_rs[f'r{r}']
        p_out_states = np.random.choice(np.arange(0, len(ai_df)), size=p, replace=False)
        #  p_out_states = np.array([p]) # this p is for 30-k_groups, leaving out data 1-by-1
        ai_df_train_rs[f'r{r}'] = ai_df.drop(p_out_states, axis=0)
        max_ai_energy_rs[f'r{r}'] = np.max(ai_df["energy"])

        train_states_rs[f'r{r}'] = np.delete(np.arange(0, len(ai_df)), p_out_states)
        val_states_rs[f'r{r}'] = p_out_states

    # Starting parameter guess

    dmd_train_rs_params = np.zeros((len(onebody_params + twobody_params), len(train_rs)))
    print("dmd_train_rs_params shape: ", dmd_train_rs_params.shape)
    E0_ind = onebody_params.index('trace')

    for i, r in enumerate(train_rs):
        ai_df = ai_df_rs[f'r{r}']
        dmd = sm.OLS(ai_df["energy"], ai_df[onebody_params + twobody_params]).fit()
        fitted_ground_state_energy = 0
        for j, param in enumerate(onebody_params + twobody_params):
            if j == E0_ind:
                continue
            dmd_train_rs_params[j][i] = dmd.params[param]
            fitted_ground_state_energy += dmd.params[param]*ai_df[param][0]

        dmd_train_rs_params[E0_ind][i] = ( ai_df["energy"][0] - fitted_ground_state_energy) / 4 # 4 is the number of sites ; make generic

    print("DMD parameters for train_rs: ", onebody_params + twobody_params)
    print(dmd_train_rs_params)

    print("Setting x0 for optimization:")

    print("Parameter function initial variables:")
    x0 = []
    x0_ind = [0]
    # Set up guess functions -------
    for j, param in enumerate(onebody_params + twobody_params):
        if param_functions[j] == 'independent':
            x0.append(dmd_train_rs_params[j])
            x0_ind.append( len(dmd_train_rs_params[j]) + x0_ind[j])
        else:
            try:
                popt, pcov = curve_fit(function_dict[param_functions[j]], train_rs, dmd_train_rs_params[j])
            except:
                sig = signature(function_dict[param_functions[j]])
                popt = np.zeros(len(sig.parameters) - 1)
            print(f"{param} : {popt}")
            x0.append(popt)
            x0_ind.append( len(popt) + x0_ind[j])
    
    x0 = np.concatenate(x0)

    print(x0, x0_ind)

    norm_rs = {}

    for r in (train_rs):
        ai_df = ai_df_rs[f'r{r}']
        ai_df_floats = ai_df.select_dtypes(include=[np.float64]) # Make sure no columns with non floats get included
        ai_var = np.var(ai_df_floats, axis=0)
        #print("Before clipping:\n", ai_var)
        des_var = ai_var.loc[ai_var.index != 'energy']
        des_var = np.clip(des_var, clip_val, np.max(des_var))  # clip so that small variances are set to 1
        ai_var.loc[ai_var.index != 'energy'] = des_var
        #print("After clipping:\n", ai_var)
        norm_rs[f'r{r}'] = 2 * ai_var

    keys = dmd.params.keys()

    # OPTMIZATION LOOP START
    print("Starting optimization")

    xmin = minimize(
        optimize_CV_para_function,
        x0,
        args=(
            x0_ind,
            param_functions,
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
            val_states_rs,
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
                                     param_functions,
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
                                     val_states_rs)

    with h5py.File(outfile, "w") as f:
        f["train_rs"] = train_rs
        f["para_w_0"] = weights[0]
        f["para_w_1"] = weights[1]
        f["loss"] = xmin.fun
        f["params"] = onebody_params + twobody_params
        f["params functions"] = param_functions
        f["params functions paramters"] = xmin.x
        f["Mean over r Spectrum RMSE - Train"] = data["Mean over r Spectrum RMSE - Train"]
        f["Mean over r Spectrum RMSE - Validation"] = data["Mean over r Spectrum RMSE - Validation"]
        f["iterations"] = xmin.nit
        f["termination_message"] = xmin.message

        for i, r in enumerate(train_rs):
            data_r = data[f"r{r}"]

            for j, k in enumerate(onebody_params + twobody_params):
                f[f"r{r}/" + "dmd_params/" + k] = dmd_train_rs_params[j][i]

            ai_df = ai_df_rs[f"r{r}"]
            f[f"r{r}/" + "ai_spectrum_range"] = np.max(ai_df["energy"]) - np.min(ai_df["energy"])

            f[f"r{r}/" + "nstates_train"] = len(train_states_rs[f"r{r}"])
            f[f"r{r}/" + "nstates_val"] = len(val_states_rs[f"r{r}"])
            f[f"r{r}/" + "state_ind_for_val"] = val_states_rs[f"r{r}"]

            for k in data_r:
                if k == "descriptors":
                    for kk in data_r[k]:
                        f[f"r{r}/" + "train/" + k + "/" + kk] = data_r[k][kk]
                elif k == "params":
                    for kk in onebody_params + twobody_params:
                        f[f"r{r}/" + "rdmd_params/" + kk] = data_r[k][kk]
                else:
                    f[f"r{r}/" + k] = data_r[k]

def inference(
    onebody: dict,
    twobody: dict,
    onebody_params: list,
    twobody_params: list,
    ai_df_rs: dict[str, pd.DataFrame],
    inference_name: str,
    nroots: int,
    outfile: str,
    matches: list,
    rs : list,
    params: dict[str, list],
    clip_val=1,
):  
    max_ai_energy_rs = {}
    norm_rs = {}
    natoms = onebody[matches[0]].shape[0]

    for r in rs:
        ai_df = ai_df_rs[f'r{r}']
        max_ai_energy_rs[f'r{r}'] = np.max(ai_df["energy"])

        ai_df_floats = ai_df.select_dtypes(include=[np.float64]) # Make sure no columns with non floats get included
        ai_var = np.var(ai_df_floats, axis=0)
        #print("Before clipping:\n", ai_var)
        des_var = ai_var.loc[ai_var.index != 'energy']
        des_var = np.clip(des_var, clip_val, np.max(des_var))  # clip so that small variances are set to 1
        ai_var.loc[ai_var.index != 'energy'] = des_var
        #print("After clipping:\n", ai_var)
        norm_rs[f'r{r}'] = 2 * ai_var

    data = {}

    print(f"Evaluating Inference natoms {natoms}")

    for r in rs:
        data[f'r{r}'] =  evaluate_loss(params[f'r{r}'],
                                    matches,
                                    [1.0, 0.0],
                                    onebody,
                                    twobody,
                                    ai_df_rs[f'r{r}'],
                                    max_ai_energy_rs[f'r{r}'],
                                    nroots,
                                    matches,
                                    None,
                                    norm_rs[f'r{r}'],
                                    r)

    with h5py.File(outfile, "a") as f:
        inf_str = f"inference_{inference_name}/"
        f[inf_str+f"rs"] = rs
        f[inf_str+f"natoms"] = natoms

        for r in rs:
            data_r = data[f"r{r}"]

            for k in data_r:
                if k == "descriptors":
                    for kk in data_r[k]:
                        f[inf_str+f"r{r}/" + "test/" + k + "/" + kk] = data_r[k][kk]
                elif k == "params":
                    for kk in onebody_params + twobody_params:
                        f[inf_str+f"r{r}/" + "rdmd_params/" + kk] = data_r[k][kk]
                else:
                    f[inf_str+f"r{r}/" + k] = data_r[k]