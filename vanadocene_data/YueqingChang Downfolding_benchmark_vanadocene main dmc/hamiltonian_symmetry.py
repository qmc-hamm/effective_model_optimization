import numpy as np
import matplotlib.pyplot as plt
from typing import List

from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

import h5py
from pyscf import fci
from scipy.spatial.transform import Rotation


def get_site_symm_ops(
    species: List[str], geom: np.ndarray, center: bool = True
) -> np.ndarray:
    """
    Grabs point group symmetry of an arrangement of atoms. Uses pymatgen to grab
    the cartesian symmetry operators for said point group. Then converts the
    cartesian symmetry operators to the site based symmetry operators where each
    site is located at each atom.

    Args:
    species: list of atom types ex: ["C", "H", "H"]
    geom: list of atom positions ex np.array([[0.,0.,0.],...])
    center: whether to center the molecule around the center of mass. This often
            will be the most symmetry.

    Returns:
    site_symm_ops : a boolean ndarray of shape [nsymmops, natoms, natoms], with
                    true where atom i is equivalent to atom j under that
                    particular symmetry operation
    """

    mol = Molecule(species, geom)
    if center:
        mol = mol.get_centered_molecule()
    mol_pga = PointGroupAnalyzer(mol)
    cart_symm_ops = mol_pga.get_symmetry_operations()
    site_symm_ops = np.zeros((len(cart_symm_ops), len(mol), len(mol)), dtype=bool)
    for k, operation in enumerate(cart_symm_ops):  # in range(len(cart_symm_ops)):
        mol_symm = mol.copy()
        mol_symm.apply_operation(operation)
        for i, m in enumerate(mol):
            for j, n in enumerate(mol_symm):
                site_symm_ops[k, i, j] = m == n

        if not np.allclose(
            site_symm_ops[k] @ site_symm_ops[k].T, np.identity(len(mol))
        ):
            raise Exception("Did not produce a unitary transformation in get_site_symm")

    return site_symm_ops


def symmetrize_onebody(O: np.ndarray, symm_ops: np.ndarray):
    """
    Performs symm_op.T @ O @ symm_op and averages for length of symm_ops.

    Args:
    O: a two-index (one-body) operator, shape [nsites, nsites]
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    O_symm : a symmetrized version of O, shape [nsites, nsites]
    """
    O_symm = np.zeros_like(O)
    for operation in symm_ops:
        O_symm += np.einsum("ai,bj, ij -> ab", operation, operation, O)
    O_symm /= symm_ops.shape[0]
    return O_symm


def random_H1(symm_ops: np.ndarray):
    """
    This function takes in a system's symmetry operators to constructs a random
    hermitian 1-body Hamiltonian.

    Args:
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Random, symmetric one-body hamiltonian with shape [nsites, nsites]
    """
    N = symm_ops.shape[1]
    H = np.random.randn(N, N)
    H = 0.5 * (H + H.T)  # Hermitian
    return symmetrize_onebody(H, symm_ops)


def onebody_symm_basis(symm_ops: np.ndarray, relative_tolerance=8):
    """
    Takes each element of one-body Hamiltonian and symmetrizes it to find
    symmetric invariant operators.
    Only the unique operators are returned.

    Args:
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Basis of one-body operators invariant to the symm_ops.
    Shape [N, nsites, nsites] where N changes based on symm_ops used.
    """
    N = symm_ops.shape[1]
    Asymm_list = []
    for i in range(N):
        for j in range(i + 1):  # Only need to go over upper triangle elements
            A = np.zeros((N, N))
            A[i, j] += 1.0
            A[j, i] += 1.0  # Normalized Hermitian
            Asymm = symmetrize_onebody(A, symm_ops)
            found = np.allclose(Asymm, np.zeros_like(Asymm))
            Asymm /= Asymm.max()
            for As in Asymm_list:
                if np.allclose(As, Asymm, rtol=relative_tolerance) or np.allclose(As, -Asymm, rtol=relative_tolerance):
                    found = True
                    break
            if not found:
                if not np.allclose(Asymm, Asymm.T):
                    raise Exception("Did not produce a Hermitian Asymm")
                Asymm_list.append(Asymm)
    return Asymm_list


def symmetrize_twobody(O: np.ndarray, symm_ops: np.ndarray):
    """
    Performs symm_op.T @ O @ symm_op and averages for length of symm_ops.

    Args:
    O: a four-index (two-body) operator, shape [nsites, nsites, nsites, nsites]
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    O_symm : a symmetrized version of O, shape [nsites, nsites, nsites, nsites]
    """
    O_symm = np.zeros_like(O)
    for operation in symm_ops:
        O_symm += np.einsum(
            "ai,bj,ck,dl, ijkl -> abcd", operation, operation, operation, operation, O
        )
    O_symm /= symm_ops.shape[0]
    return O_symm


def random_H2(symm_ops: np.ndarray):
    """
    This function takes in a system's symmetry operators to constructs a random
    hermitian 2-body Hamiltonian.

    Args:
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Random, two-body Hamiltonian invariant to the symm_ops given.
    """
    N = symm_ops.shape[1]
    H = np.random.randn(N, N, N, N)
    H = 0.5 * (H + H.T)  # Hermitian
    return symmetrize_twobody(H, symm_ops)


def twobody_symm_basis(symm_ops: np.ndarray, relative_tolerance=8):
    """
    Takes each element of two-body Hamiltonian and symmetrizes it to find
    symmetric invariant operators.
    Only the unique operators are returned.

    Args:
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Basis of two-body operators invariant to the symm_ops.
    Shape [N, nsites, nsites, nsites, nsites] where N changes based on symm_ops
    used.
    """
    N = symm_ops.shape[1]
    Asymm_list = []
    for i in range(N):
        for j in range(i + 1):  # Only need to go over upper triangle elements
            for k in range(i + 1):
                for l in range(i + 1):
                    A = np.zeros((N, N, N, N))
                    A[i, j, k, l] += 0.25  # Initial Guess
                    A[l, k, j, i] += 0.25  # Hermitian
                    A[
                        k, l, i, j
                    ] += 0.25  # Spin exchange symm (time-reversal symm for fermions)
                    A[j, i, l, k] += 0.25  # Hermitian
                    Asymm = symmetrize_twobody(A, symm_ops)
                    found = np.allclose(Asymm, np.zeros_like(Asymm))
                    Asymm /= Asymm.max()
                    for As in Asymm_list:
                        if np.allclose(As, Asymm, rtol=relative_tolerance) or np.allclose(As, -Asymm, rtol=relative_tolerance):
                            found = True
                            break
                    if not found:
                        if not np.allclose(Asymm, Asymm.T):
                            raise Exception("Did not produce a Hermitian Asymm")
                        Asymm_list.append(Asymm)
    return Asymm_list


def save_symm_term_group(fname: str, symm_terms: np.ndarray):
    """
    Args:
        fname : str
        symm_terms : 1-body or 2-body symm_terms given by onebody_symm_basis()
        or twobody_symm_basis()

    Creates hdf5 file storing basis of symmetric terms in a minimal grouping.
    Stores the number of sites, operator, indicies of sites.
    """

    with h5py.File(fname, "w") as f:
        f["minimal_groups"] = len(symm_terms)
        for i, symm_term in enumerate(symm_terms):
            index = np.array(np.where(symm_term != 0)).T
            f[f"group{i}/nsites"] = len(np.unique(index[0]))
            f[f"group{i}/operator"] = symm_term
            f[f"group{i}/indices"] = index
    return


def generate_group(
    current_list: np.ndarray,
    generators: np.ndarray,
    round: int = 9,
    iteration: int = 0,
    maximum_group_size: int = 1000,
):
    """
        The group size for a given point group are well defined and should be
        referenced via a table. The generated group size should be
        less than or equal to the point group size.
        If the generated group size is less than the point group size, it should
        be checked that it is due to basis of the symmetry operators.
        Example: s obitals symmetry operator for H4 molecule.
        H4 molecule has D4H symmetry, but the generated group will have D4
        symmetry as the s orbitals do not break sigma_h symmetry.

        Args:
        current_list: list of symmetry operators currently in the set. Should be numpy 2D arrays.
        generators: list of symmetry generators. Should be numpy 2D arrays.
        round: number of digits to round to when checking for duplicates
        interation: iteration number, CAN be used if keeping track of interation
                    or killing after certain number of iterations
        maximum_group_size: will kill recoursion after the number of symmetry
                            operators in current_list exceeds this value.
    ​
        EXAMPLE: c3v for a triangle
        generators = np.asarray( [
        [ #Identity
          [1,0,0],
          [0,1,0],
          [0,0,1],
        ],
        [ #rotation
          [0,1,0],
          [0,0,1],
          [1,0,0],
        ],
        [ #mirror
          [1,0,0],
          [0,0,1],
          [0,1,0],
        ],
        ])
        symmetry_operations = generate_group(generators, generators)
        symmetry_operations.shape[0] # group size
    """
    if current_list.shape[0] > maximum_group_size:
        raise ValueError("Group size is too large.")
    # generate new operators
    added_ops = []
    for op in current_list:
        for gen in generators:
            added_ops.append(gen @ op)
    added_ops = np.asarray(added_ops)

    # remove duplicates
    stack = np.vstack((current_list, np.asarray(added_ops)))
    new_list, new_inds = np.unique(np.round(stack, round), axis=0, return_index=True)
    new_list = stack[new_inds]

    if new_list.shape[0] == current_list.shape[0]:
        return new_list
    return generate_group(new_list, generators, iteration=iteration + 1)


def assign_twobody_names(twobody, print_nonzero=False, tol=1e-15):
    """
    returns: dictionary with classification as the key and a list of indices of the twobody terms with that classification as the value
    """
    class_terms = {}
    number = 12
    for i, term in enumerate(twobody):
        indices = np.argwhere(term)
        classification = "unknown"
        if len(np.unique(indices[0])) == 1:
            classification = "onsite"
        elif np.absolute(indices[0][0] - indices[0][1]) < tol and np.absolute(indices[0][2] - indices[0][3]) < tol:
            classification = "density"
        elif np.absolute(indices[0][0] - indices[0][3]) < tol and np.absolute(indices[0][1] - indices[0][2]) < tol:
            classification = "exchange"
        elif np.absolute(indices[0][0] - indices[0][2]) < tol and np.absolute(indices[0][1] - indices[0][3]) < tol:
            classification = "double hopping"
        if classification not in class_terms.keys():
            class_terms[classification] = []
        class_terms[classification].append((i, indices[0]))
        if print_nonzero:
            print(i, indices)
    return class_terms


def generate_rotation_p_orbitals(
    euler_vector: np.array, tol: float = 1e-4, basis_change=None
):
    """
    Rotation matrix about abritrary axis in 3D Cartesian coordinates basis:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_matrix.html
    Matrix elements in the p-orbital basis:
    http://openmopac.net/manual/rotate_atomic_orbitals_generic.html
    Basis ordering: {p_{x}, p_{y}, p_{z}}
    R in Cartesian coordinates happens to be numerically equivalent to R in {p_{x}, p_{y}, p_{z}}
    """
    if basis_change is None:
        basis_change = np.eye(3)

    symmop = Rotation.from_rotvec(euler_vector).as_matrix()
    symmop = np.einsum("ij,jk,kl->il", basis_change.T, symmop, basis_change)
    assert np.linalg.norm(np.einsum("ij,jk->ik", symmop.T, symmop) - np.eye(3)) < tol
    return symmop


def generate_rotation_d_orbitals(
    euler_vector: np.array, tol: float = 1e-4, basis_change=None
):
    """
    Rotation matrix about abritrary axis in 3D Cartesian coordinates basis:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_matrix.html
    Matrix elements in the p-orbital basis:
    http://openmopac.net/manual/rotate_atomic_orbitals_generic.html
    Basis ordering: {d_x^2-y^2, d_xz, d_z^2, d_yz, d_xy}
    """
    if basis_change is None:
        basis_change = np.eye(5)

    R = Rotation.from_rotvec(euler_vector).as_matrix()
    a, b, c = R[0,0], R[0,1], R[0,2]
    d, e, f = R[1,0], R[1,1], R[1,2]
    g, h, o = R[2,0], R[2,1], R[2,2]

    R_x2_y2_x2_y2 = (a**2 - d**2 - b**2 + e**2) / 2
    R_x2_y2_xz = a*g - b*h
    R_x2_y2_z2 = (2*g**2 - d**2 - a**2 - 2*h**2 + e**2 + b**2) / np.sqrt(12)
    R_x2_y2_yz = d*g - e*h
    R_x2_y2_xy = a*d - b*e

    R_xz_x2_y2 = a*c - d*f
    R_xz_xz = a*o + c*g
    R_xz_z2 = (2*g*o - d*f - a*c) / np.sqrt(3)
    R_xz_yz = d*o + f*g
    R_xz_xy = a*f + c*d

    R_z2_x2_y2 = (2*c**2 - b**2 - a**2 - 2*f**2 + e**2 + d**2) / np.sqrt(12)
    R_z2_xz = (2*c*o - b*h - a*g) / np.sqrt(3)
    R_z2_z2 = (4*o**2 - 2*(h**2 + g**2 + f**2 + c**2) + e**2 + d**2 + b**2 + a**2) / 6
    R_z2_yz = (2*o*f - e*h - d*g) / np.sqrt(3)
    R_z2_xy = (2*c*f - e*b - a*d) / np.sqrt(3)

    R_yz_x2_y2 = b*c - e*f
    R_yz_xz = b*o + c*h
    R_yz_z2 = (2*h*o - e*f - c*b) / np.sqrt(3)
    R_yz_yz = e*o + f*h
    R_yz_xy = b*f + c*e

    R_xy_x2_y2 = a*b - d*e
    R_xy_xz = a*h + b*g
    R_xy_z2 = (2*g*h - e*d - a*b) / np.sqrt(3)
    R_xy_yz = d*h + e*g
    R_xy_xy = a*e + b*d

    symmop = np.array(
        [
            [R_x2_y2_x2_y2, R_x2_y2_xz, R_x2_y2_z2, R_x2_y2_yz, R_x2_y2_xy],
            [R_xz_x2_y2, R_xz_xz, R_xz_z2, R_xz_yz, R_xz_xy],
            [R_z2_x2_y2, R_z2_xz, R_z2_z2, R_z2_yz, R_z2_xy],
            [R_yz_x2_y2, R_yz_xz, R_yz_z2, R_yz_yz, R_yz_xy],
            [R_xy_x2_y2, R_xy_xz, R_xy_z2, R_xy_yz, R_xy_xy],
        ]
    )
    symmop = np.einsum("ij,jk,kl->il", basis_change.T, symmop, basis_change)
    assert np.linalg.norm(np.einsum("ij,jk->ik", symmop.T, symmop) - np.eye(5)) < tol
    return symmop