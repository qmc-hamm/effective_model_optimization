import h5py
import pandas as pd
import loss_function
import os


def runCV(
    named_terms,
    ai_descriptors,
    model_descriptors,
    nroots,
    onebody_params,
    twobody_params,
    minimum_1s_occupation=3.7,
    w_0=1,
    beta=0,
    p=0,
    guess_params=None,
    state_cutoff = None
):
    """_summary_

    Parameters
    ----------
    named_terms : _type_
        _description_
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
    p : int, optional
        The number of states to leave out.
    guess_params : pd.Series, optional
        Guess parameters to override the DMD parameters if so desired. dict (keys are strings, values are floats)
        overrides the dmd parameters, by default None
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
    if state_cutoff is not None:
        ai_df = ai_df[ai_df.state < state_cutoff]
    ai_df = ai_df.reset_index()
    print("ai_df", ai_df)
    matches = onebody_params + twobody_params
    weights = [w_0, 1-w_0]

    loss_function.mapping(
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
        p,
        guess_params,
    )


def make_name(parameters):
    return "_".join(parameters[0]) + "_" + "_".join(parameters[1])

if __name__ == "__main__":
    import itertools
    parameter_sets = [
        #(['E0', 't'], ['U']),
        #(['E0', 't', 'tdiag'], ['U']),
        (['E0', 't'], ['U', 'V']),
        #(['E0', 't'], ['U', 'J']),
        #(['E0', 't'], ['U', 'V', 'J']),
        #(['E0', 't'], ['J']),

        #(['E0', 't'], ['U', 'V','J']),
        #(['E0', 't','tdiag'], ['U', 'V','J']),
    ]
    for parameters, r, state_cutoff, w0 in itertools.product(parameter_sets,
                                          ['r2.4', 'r2.8', 'r3.2', 'r3.6',  'r4.0', 'r4.4', 'r4.8'],
                                          [10],
                                          [0.6, 0.7, 0.8, 0.9, 1.0]):
        for i in range(10):
            pname = make_name(parameters)
            dirname = f"model_data_{state_cutoff}_{w0}"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            runCV(named_terms="symmetric_operators.hdf5",
                    ai_descriptors=f"ai_data/{r}.csv",
                    model_descriptors=f"{dirname}/{pname}_{i}_{r}.hdf5",
                    nroots=36,
                    onebody_params=parameters[0],
                    twobody_params=parameters[1],
                    w_0=w0,
                    beta=0,
                    p=1,
                    state_cutoff=state_cutoff,
                    )