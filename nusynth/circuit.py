from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2


def xz_rot_1q(input):
    theta, phi = input

    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.rz(phi, 0)

    return qc


def u3_gate_3q(input):
    qc = QuantumCircuit(3)
    qc.u(*input[0:3], 0) # type: ignore
    qc.u(*input[3:6], 1) # type: ignore
    qc.u(*input[6:9], 2) # type: ignore

    return qc


def su2_3q(input, n_reps):
    ansatz = EfficientSU2(
        3, reps=n_reps,
        su2_gates=['ry', 'rz']
    ).decompose().bind_parameters(input)
    qc = QuantumCircuit(3)
    qc.compose(ansatz, inplace=True)

    return qc


def qgan_3q(input, n_reps):
    qc = QuantumCircuit(3)
    for r in range(n_reps):
        pad = 3 * r
        qc.ry(input[pad], 0)
        qc.ry(input[pad + 1], 1)
        qc.ry(input[pad + 2], 2)
        qc.barrier()
        qc.cz(0, 1)
        qc.cz(1, 2)
        qc.cz(2, 0)
        qc.barrier()
    pad = 3 * n_reps
    qc.ry(input[pad], 0)
    qc.ry(input[pad + 1], 1)
    qc.ry(input[pad + 2], 2)

    return qc
