import numpy as np
from functools import reduce


#The pauli matrices plus the identity matrix.
sigma_0 = np.eye(2)
sigma_x = np.array([[0., 1.], [1., 0.]])
sigma_y = np.array([[0., 0. - 1.j], [0. + 1.j, 0.]])
sigma_z = np.array([[1., 0.],[0., -1.]])

def nkron(*args):
    """Return the kronecker product of a number of arguments, returning 1.0 by default."""
    return reduce(np.kron, args, np.array([1]))

def _check_multi_spin_operator_arg(N, operators_dict):
    idx = [*operators_dict.keys()]
    
    assert(np.min(idx) >= 1)
    assert(np.max(idx) <= N)
    assert(len(idx) <= len(np.unique(idx)))

def multi_spin_operator(N, single_site_operators):
    """
    Construct the multi-spin operator acting on an entire lattice.

    :param int N: indicating the number of sites of the Hilbert space.
    :param list(tuple) single_site_operators: of the form [(i, sigma_z), (j, sigma_x), ...] where i, j are in {1, ..., N}
    :return: numpy 2D-array of dimensions (2**N x 2**N). Returns 1 for N=0
    """
    single_site_operators_dict = dict(single_site_operators)
    _check_multi_spin_operator_arg(N, single_site_operators_dict)

    operators = [single_site_operators_dict.get(i, sigma_0) for i in range(1, N + 1)]
    
    return nkron(*operators)

def Hzz(interactions):
    N = len(interactions)
    H = np.zeros((2 ** N, 2 ** N))
    
    for i in range(0, N):
        for j in range(i + 1, N):
            J = interactions[i][j]
            H += J * multi_spin_operator(N, [(N-i, sigma_z), (N-j, sigma_z)])
    return H

def density_matrix(state_vec):
    state_vec_dagger = np.array(state_vec,ndmin=2).conj().T
    return np.kron(state_vec_dagger,state_vec)

def partial_trace(rho,N_left,N_right,left=True):
    
    N = int(np.log(rho.shape[0])/np.log(2))
    assert(N_left+N_right==N)
    
    rho_tensor = rho.reshape(2**N_left,2**N_right,2**N_left,2**N_right)
    
    if left:
        return np.trace(rho_tensor,axis1=0, axis2=2)
    else:
        return np.trace(rho_tensor,axis1=1, axis2=3)
    
def entanglement_entropy(rho,N_left,N_right,left=True):
    rho_sub = partial_trace(rho,N_left,N_right,left=left)
    
    eigenvals, eigenstates = np.linalg.eigh(rho_sub)
    eigenvals = np.maximum(eigenvals,1e-20)
    
    ent = -np.sum(eigenvals*np.log(eigenvals))
    return ent

def max_entanglement_entropy(N_left):
    return N_left*np.log(2.0)