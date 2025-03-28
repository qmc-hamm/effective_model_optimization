from pyscf import fci
import pandas as pd
import numpy as np


# This section as all the information for solving with <psi_i|rho|psi_j>
def eff_model_trans_solver(h1, h2, norb=4, nelec=(2, 2), nroots=36):
    """Solves a Effective Model Hamiltonian with an FCI approach.
    Meant to be used given operators from hamiltonian_symmetry.py.

    Parameters
    ----------
        h1 : nd.array(float, float)
            shape(norb, norb). The 1-body hamiltonian
        h2 : nd.array(float, float, float, float)
            shape(norb, norb, norb, norb). The 2-body hamiltonian
        norb : (int)
            The number of orbital sites.
        nelec : (tuple)
            Number of up and down electrons (up, down).
        nroots : (int)
            The number of eigenstates to solve for.

    Returns
    -------
        e : nd.array(float)
            Energy eigenvalues of hamiltonian. Of length nroots.
        fcivec : nd.array(nd.array(float))
            Eigenvector with each eigenvalue. Shape (nroots, ndeterminants). nderterminants is the number of
            determinants needed to describe wavefunctions.
        n_rdm1s : nd.array()
            Shape (nroots, 2, norbs, norbs). Spin seperated ((up,up), (down,down)) 1-body reduced density matricies.
        n_rdm2s : nd.array()
            Shape (nroots, 3, norbs, norbs, norbs, norbs).
            Spin seperated ((up,up,up,up), (up,up,down,down), (down,down,down,down)) 2-body reduced density matricies.
    """
    e, fcivec = fci.direct_spin1.kernel(
        h1, h2, norb, nelec, nroots=nroots, max_space=30, max_cycle=100, orbsym=None
    )

    n_rdm1s = np.zeros((nroots, nroots, 2, norb, norb))
    n_rdm2s = np.zeros((nroots, nroots, 4, norb, norb, norb, norb))
    for root1 in range(nroots):
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
    """Constructs the hamiltonian to be solved and once solved compiles relavant descriptors.

    Parameters
    ----------
    onebody : dict
        Dictionary of the 1-body operators. Each operator of the shape (norb,norb).
    twobody : dict
        Dictionary of the 2-body operators. Each operator of the shape (norb,norb,norb,norb).
    parameters : list
        The parameters to terms to set in the Hamiltonian, should match the keys in onebody and twobody.
    nroots : int
        Number of eigenstates to solver for the hamiltonian.

    Returns
    -------
    pd.DataFrame
        Provides the descriptors of the eigenstates, <operator>, examples being energy, the <onebody>, and <twobody>.
    """

    norb = 4  # should be passed in
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb, norb, norb, norb))
    for k in onebody.keys():
        if k in parameters.keys():
            h1 += parameters[k] * onebody[k]
    for k in twobody.keys():
        if k in parameters.keys():
            h2 += parameters[k] * twobody[k]

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
    Meant to be used given operators from hamiltonian_symmetry.py.

    Parameters
    ----------
        h1 : nd.array(float, float)
            shape(norb, norb). The 1-body hamiltonian
        h2 : nd.array(float, float, float, float)
            shape(norb, norb, norb, norb). The 2-body hamiltonian
        norb : (int)
            The number of orbital sites.
        nelec : (tuple)
            Number of up and down electrons (up, down).
        nroots : (int)
            The number of eigenstates to solve for.

    Returns
    -------
        e : nd.array(float)
            Energy eigenvalues of hamiltonian. Of length nroots.
        fcivec : nd.array(nd.array(float))
            Eigenvector with each eigenvalue. Shape (nroots, ndeterminants). nderterminants is the number of
            determinants needed to describe wavefunctions.
        n_rdm1s : nd.array()
            Shape (nroots, 2, norbs, norbs). Spin seperated ((up,up), (down,down)) 1-body reduced density matricies.
        n_rdm2s : nd.array()
            Shape (nroots, 3, norbs, norbs, norbs, norbs).
            Spin seperated ((up,up,up,up), (up,up,down,down), (down,down,down,down)) 2-body reduced density matricies.
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
        davidson_only=False,  # TODO: compare timing to fci solver, once davidson solver bug in pyscf is solved
        ci0=ci0,
    )  # These davidson_only and ci0 can help speed up the solver

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
    """_summary_

    Parameters
    ----------
    onebody : dict
        Dictionary of the 1-body operators. Each operator of the shape (norb,norb).
    twobody : dict
        Dictionary of the 2-body operators. Each operator of the shape (norb,norb,norb,norb).
    parameters : list
        The parameters to terms to set in the Hamiltonian, should match the keys in onebody and twobody.
    nroots : int
        Number of eigenstates to solver for the hamiltonian.

    Returns
    -------
    pd.DataFrame
        Provides the descriptors of the eigenstates, <operator>, examples being energy, the <onebody>, and <twobody>.
    """
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

    descriptors = {}
    descriptors["energy"] = h_eff_energies
    for k, it in onebody.items():
        descriptors[k] = np.einsum("mn,ismn -> i", it, rdm1)
    for k, it in twobody.items():
        descriptors[k] = np.einsum(
            "mnop,ismnop -> i", it, rdm2[:, [0, 1, 3], :, :, :, :]
        )
    return descriptors, h_eff_fcivec


def test():
    """
    put something here to test the code. Maybe hubbard model and tb model example. Similar to pyscf's example tests for fci.
    """


if __name__ == "__main__":
    test()
