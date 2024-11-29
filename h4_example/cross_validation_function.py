import h5py
import pandas as pd
import loss_function_para_function
import os


def runCV(named_terms,
          ai_dir,
          model_descriptors,
          nroots,
          onebody_params,
          twobody_params,
          rs,
          minimum_1s_occupation=3.7,
          w0=1,
          beta=0,
          p=0,
          guess_params=None,
          state_cutoff=None
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

    ai_df_rs = {}
    for r in rs:
        ai_df = pd.read_csv(ai_dir + f"r{r}.csv")
        ai_df = ai_df[ai_df.E0 > minimum_1s_occupation]
        ai_df = ai_df[ai_df.U < 1.3]  # Need to remove the top two states from the optimization
        if state_cutoff is not None:
            ai_df = ai_df[ai_df.state < state_cutoff]
        ai_df = ai_df.reset_index()
        ai_df_rs[f'r{r}'] = ai_df

    #print(ai_df_rs)

    matches = onebody_params + twobody_params
    weights = [w0, 1 - w0]

    loss_function_para_function.mapping(
        onebody,
        twobody,
        onebody_params,
        twobody_params,
        ai_df_rs,
        nroots,
        model_descriptors,
        matches,
        rs,
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
        (['E0', 't'], ['U']),
        #(['E0', 't', 'tdiag'], ['U']),
        #(['E0', 't'], ['U', 'V']),
        #(['E0', 't'], ['U', 'J']),
        #(['E0', 't'], ['U', 'V', 'J']),
        #(['E0', 't'], ['J']),

        #(['E0', 't'], ['U', 'V','J']),
        #(['E0', 't','tdiag'], ['U', 'V','J']),
    ]
    rs_sets = [
        [2.2, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.0, 6.0, 7.0]
    ]
    for parameters, rs, state_cutoff, w0 in itertools.product(parameter_sets,
                                                             rs_sets,
                                                             [10],
                                                             [0.9]):
        for i in range(10):
            pname = make_name(parameters)
            dirname = f"func_model_data_{state_cutoff}_{w0}"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            runCV(named_terms="symmetric_operators.hdf5",
                  ai_dir="ai_data/",
                  model_descriptors=f"{dirname}/{pname}_{i}.hdf5",
                  nroots=36,
                  onebody_params=parameters[0],
                  twobody_params=parameters[1],
                  rs=rs,
                  w0=w0,
                  beta=0,
                  p=1,
                  state_cutoff=state_cutoff,
                  )
