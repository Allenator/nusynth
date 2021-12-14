import numpy as np
from qgd_python.decomposition.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition as qgd
from qiskit import transpile
from qiskit.circuit.library import UGate


n_layers_dict = {
    2: 3,
    3: 17,
    4: 77
}


def decompose(
    unitary,
    optimize_layer_num=False,
    max_n_layers_dict=None,
    initial_guess='zeros',
    tolerance=None, verbose=False
):
    dec = qgd(
        unitary.T.conj(),
        optimize_layer_num=optimize_layer_num,
        initial_guess=initial_guess
    )
    if max_n_layers_dict is not None:
        dec.set_Max_Layer_Num(max_n_layers_dict)
    if tolerance is not None:
        dec.set_Optimization_Tolerance(tolerance)
    dec.set_Verbose(verbose)
    dec.Start_Decomposition(finalize_decomp=True, prepare_export=True)
    qc = transpile(dec.get_Quantum_Circuit(), optimization_level=0)

    return qc


def assert_n_qubits(n_qubits):
    assert n_qubits in [2, 3, 4]


def n_layers(n_qubits):
    assert_n_qubits(n_qubits)

    return n_layers_dict[n_qubits]


def n_ugates(n_qubits):
    return n_layers(n_qubits) * 2 + n_qubits


def n_params(n_qubits):
    return n_ugates(n_qubits) * 2 + n_qubits


def postprocess(circuit, normalize=False):
    n_qubits = len(circuit.qubits)
    thresh = n_ugates(n_qubits) - n_qubits
    params = np.empty(n_params(n_qubits))

    for i, gate in enumerate(
        [g for g, _, _ in circuit.data if type(g) is UGate]
    ):
        gp = gate.params
        if i < thresh:
            idx = 2 * i
            params[idx] = gp[0]
            params[idx + 1] = gp[2]
        else:
            idx = 2 * thresh + 3 * (i - thresh)
            params[idx:idx + 3] = gp

    if normalize:
        return params % (2 * np.pi)
    else:
        return params
