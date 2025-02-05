import numpy as np


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