import numpy as np
from scipy.linalg import logm


def _assert_positive_definite(p: np.ndarray, q: np.ndarray) -> None:
    """Assert that two matrices are positive definite."""
    assert np.all(np.linalg.eigvals(p) > 0) and np.all(np.linalg.eigvals(q) > 0), "Matrices must be positive definite"

def log_det_divergence(p:np.ndarray, q:np.ndarray) -> float:
    r"""Calculate log determinant divergence between two positive definite matrices p and q.
    
    .. math::
        D_{LD}(p, q) = \text{tr}(p q^{-1}) - \log(\det(p q^{-1})) - n

    Parameters
    ----------
    p : np.ndarray
        First positive definite matrix.
    q : np.ndarray
        Second positive definite matrix.
    
    Returns
    -------
    float
        Log determinant divergence between p and q.
    """
    _assert_positive_definite(p, q)  # Ensure the matrices are positive definite
    q_inv = np.linalg.inv(q)  
    pq_inv = p @ q_inv  
    trace_term = np.trace(pq_inv)
    _, log_det_term = np.linalg.slogdet(pq_inv) 
    n = p.shape[0]  # Get the dimensionality
    return trace_term - log_det_term - n


def von_neuman_divergence(p:np.ndarray, q:np.ndarray) -> float:
    r"""Calculate Von Neumann divergence between two positive definite matrices p and q.

    .. math::
        D_{VN}(p, q) = \text{tr}(p (\log(p) - \log(q))) + \text{tr}(-p + q)

    Parameters
    ----------
    p : np.ndarray
        First positive definite matrix.
    q : np.ndarray
        Second positive definite matrix.
    
    Returns
    -------
    float
        Von Neumann divergence between p and q.
    """
    _assert_positive_definite(p, q)  # Ensure the matrices are positive definite
    log_p = logm(p)  # Compute the matrix logarithm of p
    log_q = logm(q)  # Compute the matrix logarithm of q
    first_term = np.trace(p @ (log_p - log_q))  
    second_term = np.trace(-p+q)
    return first_term + second_term