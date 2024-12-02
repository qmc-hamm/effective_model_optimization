# About Project

Repository for the latest workflow to create an effective model Hamiltonians with uncertainty analysis.

### Folders
- src → latest workflow
- plots → example plots
- h4_example → example workflow with data
- hchain_yueqing → hydrogen chain data set compiled by Yueqing, effective model workflow setup in progress
- vanadocene_data → hydrogen chain data set compiled by Yueqing, effective model workflow setup in progress


# How to run

Inside `h4_example` run:

``` python cross_validation_function.py ```

### Hyperparameters
> parameter_sets : Defines the model physics
> rs_sets : which r(configuration) data is used for training and validation. The leftover data can be used for testing later.
> state_cutoff : Which ab initio state to cut off. At the lower r values, there is less ab intio data.
> w0s : The ratio that the loss function values matching the energies of the states vs the descriptors of the states.
