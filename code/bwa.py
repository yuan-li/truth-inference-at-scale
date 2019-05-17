import numpy as np
import scipy.sparse as ssp

def bwa_binary(y_exists_ij, y_is_one_ij, W_i, lambda_, a_v, adj_coef):
    N_j = y_exists_ij.sum(axis=0)
    z_i = y_is_one_ij.sum(axis=-1) / y_exists_ij.sum(axis=-1)
    
    b_v = a_v * W_i.dot(np.multiply(z_i, 1-z_i)) / y_exists_ij.sum() * adj_coef
    for _ in range(500):
        last_z_i = z_i.copy()
        
        mu  = z_i.mean()
        v_j = (a_v+N_j) / (b_v + (y_exists_ij.multiply(z_i)-y_is_one_ij).power(2).sum(0))
        z_i = (lambda_*mu + y_is_one_ij.dot(v_j.T)) / (lambda_ + y_exists_ij.dot(v_j.T))
        
        if np.allclose(last_z_i, z_i, rtol=1e-3): break
    return z_i.A1

def bwa(tuples, a_v=15, lambda_=1, prior_correction=True):
    num_items, num_workers, num_classes = tuples.max(axis=0) + 1
    num_labels = tuples.shape[0]
    W_i = np.bincount(tuples[:, 0])
    
    adj_coef = 4 * (1 - 1 / num_classes) if prior_correction else 1
    
    y_exists_ij = ssp.coo_matrix((np.ones(num_labels), tuples[:, :2].T),
                                 shape=(num_items, num_workers), dtype=np.bool).tocsr()
    y_is_one_kij = []
    for k in range(num_classes):
        selected = (tuples[:, 2] == k)
        y_is_one_kij.append(ssp.coo_matrix((np.ones(selected.sum()), tuples[selected, :2].T),
                                           shape=(num_items, num_workers), dtype=np.bool).tocsr())
    z_ik = np.empty((num_items, num_classes))
    for k in range(num_classes):
        z_ik[:, k] = bwa_binary(y_exists_ij, y_is_one_kij[k], W_i, lambda_, a_v, adj_coef)
    return z_ik
