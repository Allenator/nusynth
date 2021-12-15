import contextlib
import math

import joblib
import numpy as np
from qiskit import Aer
from scipy.stats import unitary_group
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/49950707
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# https://scicomp.stackexchange.com/questions/10748/cartesian-products-in-numpy
def cross_product(x, y):
    cross_product = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return cross_product


def random_unitary(n_qubits):
    return unitary_group.rvs(2 ** n_qubits)


def unitary_to_vec(unitary):
    assert unitary.shape[0] == unitary.shape[1]
    unitary = unitary.flatten()

    return np.concatenate((np.real(unitary), np.imag(unitary)))


def vec_to_unitary(vec):
    dim = int(math.sqrt(len(vec) / 2))
    assert dim ** 2 * 2 == len(vec)

    vec = vec.reshape((2, -1))
    unitary = np.apply_along_axis(lambda args: [complex(*args)], 0, vec)

    return unitary.reshape((dim, dim))


def circuit_to_unitary(circuit):
    res = Aer.get_backend('unitary_simulator').run(circuit).result()
    return res.get_unitary(circuit)


def pca(main, aux_list, n_components=2):
    ss = StandardScaler()
    pca = PCA(n_components=n_components)
    main_r = ss.fit_transform(main)
    main_pcs = pca.fit_transform(main_r)

    aux_pcs_list = [
        pca.transform(ss.transform(aux))
        for aux in aux_list
    ]

    return main_pcs, aux_pcs_list
