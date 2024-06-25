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


def test_lamb_with_lots_parameters():
    onebody_params = ["E0", "t"]
    twobody_params = ["U",'V', 'J','J_diag','hop_hop', 'hop_hop2', 'ni_hop', 'ni_hop2']  
    set_up_h4(
        "../h4_data/named_terms_new.hdf5",
        "../h4_data/ai_descriptors.csv",
        "model_output.hdf5",
        nroots=36, 
        onebody_params=onebody_params,
        twobody_params=twobody_params,
        minimum_1s_occupation=3.7,
        w_0=0.6,
        beta=4.0,
        lamb=1.0,
    )


if __name__=="__main__":
    test_lamb_with_lots_parameters()