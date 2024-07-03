import h5py
import pandas as pd
import loss_function


def set_up_h4(
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
    lamb : int, optional
        The penalty loss for parameters, by default 0
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
    ai_df = ai_df[ ai_df.U < 1.3]  # Need to remove the top two states from the optimization

    matches = onebody_keys[1:] + twobody_keys  # ['t', 'U' , 'ni_hop'] #'V'

    w_1 = 1 - w_0

    weights = [w_0, w_1, lamb]  # w_0, w_1, lamb

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
        guess_params,
    )


def run_tu():
    """Run optimization of the TU model.
    """
    onebody_params = ["E0", "t"]
    twobody_params = ["U"]  
    set_up_h4(
        "../h4_data/named_terms_new.hdf5",
        "../h4_data/ai_descriptors.csv",
        "model_output.hdf5",
        nroots=36, # (4 sites choose 2 electrons)^2 , ^2 is for both spin up and spin down
        onebody_params=onebody_params,
        twobody_params=twobody_params,
        minimum_1s_occupation=3.7,
        w_0=0.6,
        beta=4.0,
        lamb=0,
    )


def test_lamb_with_lots_parameters(lamb=0.0, guess_params=None):
    """Testing using lambda penalty loss for the solver.
    """
    onebody_params = ["E0", "t"]
    twobody_params = ["U",'V', 'J','J_diag','hop_hop', 'hop_hop2', 'ni_hop', 'ni_hop2']  
    set_up_h4(
        "../h4_data/named_terms_new.hdf5",
        "../h4_data/ai_descriptors.csv",
        f"model_output_{lamb}.hdf5",
        nroots=36, 
        onebody_params=onebody_params,
        twobody_params=twobody_params,
        minimum_1s_occupation=3.7,
        w_0=0.6,
        beta=4.0,
        lamb=lamb,
        guess_params=guess_params,
    )

def test_lamb_all_parameters(lamb=0.0, guess_params=None):
    """Testing using lambda penalty loss for the solver.
    """
    onebody_params = ["E0", "t", "tdiag"]
    twobody_params = ["U",'V', "Vdiag", 
                      "J","J_diag",
                      "hop_hop", "hop_hop2", "hop_hop3", "hop_hop4","hop_hop5",
                      "hop_hopdiag", "hop_hopdiag2", "hopdiag_hopdiag", "hopdiag_hopdiag2",
                      "ni_hop", "ni_hop2", "ni_hop3", "ni_hopdiag"]  
    set_up_h4(
        "../h4_data/named_terms_new.hdf5",
        "../h4_data/ai_descriptors.csv",
        f"model_output_{lamb}.hdf5",
        nroots=36, 
        onebody_params=onebody_params,
        twobody_params=twobody_params,
        minimum_1s_occupation=3.7,
        w_0=0.6,
        beta=4.0,
        lamb=lamb,
        guess_params=guess_params,
    )


if __name__=="__main__":
    #run_tu()
    test_lamb_all_parameters(lamb=0.1)