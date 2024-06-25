from pyscf import fci
import pandas as pd
import numpy as np


# This section as all the information for solving with <psi_i|rho|psi_j>
def eff_model_trans_solver(h1, h2, norb=4, nelec=(2, 2), nroots=36):
    """Solves a Effective Model Hamiltonian with an FCI approach.
    Meant to be used in with hamiltonian_symmetry.py .
    """
    e, fcivec = fci.direct_spin1.kernel(
        h1, h2, norb, nelec, nroots=nroots, max_space=30, max_cycle=100, orbsym=None
    )

    n_rdm1s = np.zeros((nroots, nroots, 2, norb, norb))
    n_rdm2s = np.zeros((nroots, nroots, 4, norb, norb, norb, norb))
    for root1 in range(nroots):
        # rdm1s, rdm2s =  fci.direct_spin1.make_rdm12s(fcivec[root1], norb=norb, nelec=nelec)
        for root2 in range(nroots):
            rdm1s, rdm2s = fci.direct_spin1.trans_rdm12s(
                fcivec[root1], fcivec[root2], norb=norb, nelec=nelec
            )
            for i, r in enumerate(rdm1s):
                n_rdm1s[root1, root2, i, :, :] = r
            for i, r in enumerate(rdm2s):
                n_rdm2s[root1, root2, i, ...] = r
    return e, fcivec, n_rdm1s, n_rdm2s


def solve_effective_trans_hamiltonian(
    onebody: dict, twobody: dict, parameters: list, nroots: int
) -> pd.DataFrame:
    norb = 4  # should be passed in
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb, norb, norb, norb))
    for k in onebody.keys():
        if k in parameters.keys():
            h1 += parameters[k] * onebody[k]
    for k in twobody.keys():
        if k in parameters.keys():
            h2 += parameters[k] * twobody[k]
            # Changed
    h_eff_energies, h_eff_fcivec, rdm1, rdm2 = eff_model_trans_solver(
        h1, h2, norb=norb, nelec=(2, 2), nroots=nroots
    )

    print("lengths of fcivec ", len(h_eff_energies), len(h_eff_fcivec[0]))

    descriptors = {}
    descriptors["energy"] = np.diag(h_eff_energies)
    for k, it in onebody.items():
        print(it.shape, rdm1.shape)
        descriptors[k] = np.einsum("mn,ijsmn -> ij", it, rdm1)
    for k, it in twobody.items():
        descriptors[k] = np.einsum(
            "mnop,ijsmnop -> ij", it, rdm2[:, :, [0, 1, 3], :, :, :, :]
        )
    return descriptors


# This section is only <psi_i|pho|psi_i>, computationally faster
def eff_model_solver(h1, h2, norb=4, nelec=(2, 2), nroots=36, ci0: np.ndarray = None):
    """Solves a Effective Model Hamiltonian with an FCI approach.
    Meant to be used in with hamiltonian_symmetry.py .
    """
    e, fcivec = fci.direct_spin1.kernel(
        h1,
        h2,
        norb,
        nelec,
        nroots=nroots,
        max_space=30,
        max_cycle=100,
        orbsym=None,
        davidson_only=False, # TODO: compare timing to fci solver, once davidson solver bug in pyscf is solved 
        ci0=ci0,
    )  # These two can help speed up the solver

    n_rdm1s = np.zeros((nroots, 2, norb, norb))
    n_rdm2s = np.zeros((nroots, 4, norb, norb, norb, norb))
    for root1 in range(nroots):
        rdm1s, rdm2s = fci.direct_spin1.make_rdm12s(
            fcivec[root1], norb=norb, nelec=nelec
        )
        for i, r in enumerate(rdm1s):
            n_rdm1s[root1, i, :, :] = r
        for i, r in enumerate(rdm2s):
            n_rdm2s[root1, i, ...] = r
    return e, fcivec, n_rdm1s, n_rdm2s


def solve_effective_hamiltonian(
    onebody: dict, twobody: dict, parameters: list, nroots: int, ci0: np.ndarray = None
) -> pd.DataFrame:
    norb = 4
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb, norb, norb, norb))
    for k in onebody.keys():
        if k in parameters.keys():
            h1 += parameters[k] * onebody[k]
    for k in twobody.keys():
        if k in parameters.keys():
            h2 += parameters[k] * twobody[k]

    h_eff_energies, h_eff_fcivec, rdm1, rdm2 = eff_model_solver(
        h1, h2, norb=norb, nelec=(2, 2), nroots=nroots, ci0=ci0
    )

    # print('lengths of fcivec ',len(h_eff_energies), len(h_eff_fcivec[0]))

    descriptors = {}
    descriptors["energy"] = h_eff_energies
    for k, it in onebody.items():
        # print(it.shape, rdm1.shape)
        descriptors[k] = np.einsum("mn,ismn -> i", it, rdm1)
    for k, it in twobody.items():
        descriptors[k] = np.einsum(
            "mnop,ismnop -> i", it, rdm2[:, [0, 1, 3], :, :, :, :]
        )
    return descriptors, h_eff_fcivec


def test():
    """
    put something here to test the code
    """


if __name__ == "__main__":
    test()
