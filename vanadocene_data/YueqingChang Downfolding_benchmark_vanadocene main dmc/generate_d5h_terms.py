import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from hamiltonian_symmetry import (
    generate_group,
    generate_rotation_d_orbitals,
    onebody_symm_basis,
    twobody_symm_basis,
    assign_twobody_names,
)

def generate_d5h_group():
    # generate_rotation_d_orbitals() generates symmop matrix in the basis: {d_x^2-y^2, d_xz, d_z^2, d_yz, d_xy}
    # We want symmops in the basis: {d_z^2, d_xy, d_xz, d_yz, d_x^2-y^2}
    mapping = [2, 4, 1, 3, 0]
    euler_vector = (2 * np.pi / 5) * np.array([0, 0, 1])
    C5 = generate_rotation_d_orbitals(euler_vector)
    C5 = C5[mapping, :]
    C5 = C5[:, mapping]

    C2_prime = np.diag([ 1,-1, 1,-1, 1])
    sigma_h  = np.diag([ 1, 1,-1,-1, 1])

    #plot_symmop(C3, title="C3")
    #plot_symmop(sigma_yz, title="sigma_yz")
    symmops = np.stack((C5, C2_prime, sigma_h), axis=0)
    symmops = generate_group(symmops, symmops)
    #verify_c3v_rotations(symmops)
    print(f"Generated symmops: {symmops.shape[0]} unique matrices")
    return symmops

def generate_symmetry_invariant_terms(
    symmops,
    tol=1e-4,
    relative_tolerance=25,
    print_two_body=True,
):
    one_body = onebody_symm_basis(symmops)
    one_body = reduce_to_minimal_set(one_body)
    print(f"Number of 1-body terms = {len(one_body)}")
    for term in one_body:
        print(f"1-body term \n {np.argwhere(np.absolute(term) > tol)}")
    two_body = twobody_symm_basis(symmops, relative_tolerance=relative_tolerance)
    two_body = reduce_to_minimal_set(two_body)
    if print_two_body:
        print(f"Number of 2-body terms = {len(two_body)}")
        print(assign_twobody_names(two_body))
    for two_body_op in two_body:
        verify_symmetry(two_body_op, symmops)
    return one_body, two_body

def reduce_to_minimal_set(operators):
    operators = np.array(operators)
    if len(operators.shape) == 3:
        A = np.zeros((operators.shape[1] ** 2, operators.shape[0]))
    elif len(operators.shape) == 5:
        A = np.zeros((operators.shape[1] ** 4, operators.shape[0]))
    for i, term in enumerate(operators):
        A[:, i] = term.reshape(-1)
    A = pick_linearly_indepedent(A)
    operators_new = []
    for i in range(A.shape[1]):
        if len(operators.shape) == 3:
            operators_new.append(
                A[:, i].reshape(operators[0].shape[1], operators[0].shape[1])
            )
        elif len(operators.shape) == 5:
            operators_new.append(
                A[:, i].reshape(
                    operators[0].shape[1],
                    operators[0].shape[1],
                    operators[0].shape[1],
                    operators[0].shape[1],
                )
            )
    return operators_new

def pick_linearly_indepedent(A, tol=1e-4):
    S = np.linalg.svd(A, full_matrices=False)[1]
    unnecessary_columns = np.nonzero(S < tol)[0]
    A_new = np.copy(A)
    if len(unnecessary_columns) > 0:
        print(
            f"{len(unnecessary_columns)} zero singular values detected out of {A.shape[1]}"
        )
        i = 0
        while i < 50:
            columns_to_select = np.random.choice(
                range(A.shape[1]),
                size=A.shape[1] - len(unnecessary_columns),
                replace=False,
            )
            A_new = A[:, columns_to_select]
            if np.linalg.matrix_rank(A_new) == A_new.shape[1]:
                break
            i += 1
    return A_new

def verify_symmetry(two_body_op, symmops, invariance_tol=1e-3):
    for R in symmops:
        two_body_op_transformed = np.einsum(
            "ijkl,ai,bj,ck,dl->abcd", two_body_op, R, R, R, R
        )
        assert np.linalg.norm(two_body_op_transformed - two_body_op) < invariance_tol

if __name__ == "__main__":
    symmops_d5h = generate_d5h_group()

    one_body_d5h, two_body_d5h = generate_symmetry_invariant_terms(symmops_d5h)

    with h5py.File("d5h_symmetry_allowed_terms.hdf5", "w") as f:
        f['a_mapping_order_for_d5h'] = '{d_z^2, d_xy, d_xz, d_yz, d_x^2-y^2}'
        for i in range(len(one_body_d5h)):
            f[f"one_body/{i}"] = one_body_d5h[i]
        for i in range(len(two_body_d5h)):
            f[f"two_body/{i}"] = two_body_d5h[i]