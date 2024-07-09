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
    p=0,
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
    ai_df = ai_df[ ai_df.U < 1.3]  # Need to remove the top two states from the optimization

    matches = onebody_params + twobody_params

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
        p, 
        guess_params,
    )


def run_tu(p=0):
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
        beta=0.0,
        lamb=0,
        p=p, 
    )

def CV_run_tu(p=1, nCV=1):
    """Run optimization of the TU model. nCV is number of Cross Validation Runs.
    """
    onebody_params = ["E0", "t"]
    twobody_params = ["U"]  

    for i in range(nCV):
        set_up_h4(
            "../h4_data/named_terms_new.hdf5",
            "../h4_data/ai_descriptors.csv",
            f"../CVmodels/CV{i}_model_output.hdf5",
            nroots=36, # (4 sites choose 2 electrons)^2 , ^2 is for both spin up and spin down
            onebody_params=onebody_params,
            twobody_params=twobody_params,
            minimum_1s_occupation=3.7,
            w_0=0.9,
            beta=0.0,
            lamb=0,
            p=p, 
        )

def CV_run_tuv(p=1, nCV=1):
    """Run optimization of the TU model. nCV is number of Cross Validation Runs.
    """
    onebody_params = ["E0", "t"]
    twobody_params = ["U", "V"]  

    for i in range(nCV):
        set_up_h4(
            "../h4_data/named_terms_new.hdf5",
            "../h4_data/ai_descriptors.csv",
            f"../CVmodels/CV{i}_model_output.hdf5",
            nroots=36, # (4 sites choose 2 electrons)^2 , ^2 is for both spin up and spin down
            onebody_params=onebody_params,
            twobody_params=twobody_params,
            minimum_1s_occupation=3.7,
            w_0=0.9,
            beta=0.0,
            lamb=0,
            p=p, 
        )

def CV_run_tuw(p=1, nCV=1):
    """Run optimization of the TU model. nCV is number of Cross Validation Runs.
    """
    onebody_params = ["E0", "t"]
    twobody_params = ["U", "ni_hop"]  

    for i in range(nCV):
        set_up_h4(
            "../h4_data/named_terms_new.hdf5",
            "../h4_data/ai_descriptors.csv",
            f"../CVmodels/t-U-W/CV{i}_model_output.hdf5",
            nroots=36, # (4 sites choose 2 electrons)^2 , ^2 is for both spin up and spin down
            onebody_params=onebody_params,
            twobody_params=twobody_params,
            minimum_1s_occupation=3.7,
            w_0=0.9,
            beta=0.0,
            lamb=0,
            p=p, 
        )

def CV_run_tuj(p=1, nCV=1):
    """Run optimization of the TU model. nCV is number of Cross Validation Runs.
    """
    onebody_params = ["E0", "t"]
    twobody_params = ["U", "J"]  

    for i in range(nCV):
        set_up_h4(
            "../h4_data/named_terms_new.hdf5",
            "../h4_data/ai_descriptors.csv",
            f"../CVmodels/t-U-J/CV{i}_model_output.hdf5",
            nroots=36, # (4 sites choose 2 electrons)^2 , ^2 is for both spin up and spin down
            onebody_params=onebody_params,
            twobody_params=twobody_params,
            minimum_1s_occupation=3.7,
            w_0=0.9,
            beta=0.0,
            lamb=0,
            p=p, 
        )

if __name__=="__main__":
    CV_run_tuw(p=5, nCV=10)
    CV_run_tuj(p=5, nCV=10)