import numpy as np
from qgd_python.decomposition.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition as qgd
from qiskit import transpile
from qiskit.circuit.library import Barrier


default_n_layers_dict = {
    2: 4,
    3: 16,
    4: 60
}


def decompose(
    unitary,
    optimize_layer_num=False,
    n_layers_dict=default_n_layers_dict,
    initial_guess='zeros',
    tolerance=None,
    reformat=True,
    verbose=False
):
    dec = qgd(
        unitary.T.conj(),
        optimize_layer_num=optimize_layer_num,
        initial_guess=initial_guess
    )
    n_layers_dict = default_n_layers_dict | n_layers_dict
    dec.set_Max_Layer_Num(n_layers_dict)
    if tolerance is not None:
        dec.set_Optimization_Tolerance(tolerance)
    dec.set_Verbose(verbose)
    dec.Start_Decomposition(finalize_decomp=True, prepare_export=True)
    qc = dec.get_Quantum_Circuit()

    if reformat:
        return reformat_circuit(qc)
    else:
        return qc


def extract_params(circuit, n_layers_dict, normalize=False):
    n_qubits = len(circuit.qubits)
    params = np.empty(n_params(n_qubits, n_layers_dict))

    real_n_layers_dict = {i_q + 1: 0 for i_q in range(1, n_qubits)}
    u_params = []

    for op, qa, _ in circuit.data:
        for i_q in range(1, n_qubits):
            if op.name == 'cx' and qa[0].index == i_q and qa[1].index == 0:
                real_n_layers_dict[i_q + 1] += i_q
        if op.name == 'u':
            u_params.extend(op.params)

    idx = 3 * n_qubits # initial rotations
    params[:idx] = u_params[:idx]

    for q in range(n_qubits, 1, -1):
        real_nl = real_n_layers_dict[q]
        spec_nl = n_layers_dict[q]
        assert real_nl == 0 or real_nl == spec_nl
        n_q_params = 3 * 2 * spec_nl

        if real_nl == 0:
            params[idx:idx + n_q_params] = 0
        elif real_nl == spec_nl:
            params[idx:idx + n_q_params] = u_params[idx:idx + n_q_params]
        else:
            raise ValueError(
                f'Observed {real_nl} {q}-qubit layers, '
                f'inconsistent with {spec_nl} layers specified'
            )

        idx += n_q_params

    assert idx == n_params(n_qubits, n_layers_dict)

    if normalize:
        return params % (2 * np.pi)
    else:
        return params


def assert_n_qubits(n_qubits):
    assert n_qubits in [2, 3, 4]


def assert_n_layers_dict(n_qubits, n_layers_dict):
    for q in range(2, n_qubits + 1):
        assert q in n_layers_dict
        assert n_layers_dict[q] % ((q - 1) * 2) == 0


def n_params(n_qubits, n_layers_dict):
    n_u = n_qubits # initial rotations
    for q in range(2, n_qubits + 1):
        n_u += n_layers_dict[q] * 2
    return n_u * 3


def t0(circuit):
    return transpile(circuit, optimization_level=0)


def reformat_circuit(circuit):
    circuit = _reverse_circuit(circuit)
    circuit = t0(circuit)
    circuit = _insert_barriers(circuit)
    circuit = _reverse_circuit(circuit)
    circuit = t0(circuit)

    return circuit


def _reverse_circuit(circuit):
    circuit = circuit.copy()
    n_qubits = len(circuit.qubits)

    for _, qa, _ in circuit.data:
        for i, a in enumerate(qa):
            qa[i] = circuit.qubits[n_qubits - a.index - 1]

    return circuit


def _get_last_op_pos(circuit):
    last_cx_pos_dict = {}
    last_u_pos_dict = {}

    for i, (op, qa, _) in enumerate(circuit.data):
        if op.name == 'u':
            last_u_pos_dict[qa[0].index] = i
        if op.name == 'cx':
            last_cx_pos_dict[(qa[0].index, qa[1].index)] = i

    return last_u_pos_dict, last_cx_pos_dict


def _insert_barriers(circuit):
    circuit = circuit.copy()
    n_qubits = len(circuit.qubits)
    _, last_cx_pos_dict = _get_last_op_pos(circuit)

    n_barriers = 0
    for q0 in range(n_qubits - 1):
        q1 = q0 + 1
        if (q0, q1) in last_cx_pos_dict:
            pos = last_cx_pos_dict[(q0, q1)] + (n_qubits + 1 - q0) + n_barriers
            circuit.data.insert(
                pos,
                (Barrier(n_qubits), circuit.qubits, [])
            )
            n_barriers += 1

    return circuit
