from qiskit.quantum_info import process_fidelity
from qiskit.quantum_info.operators import Operator

import nusynth.utils as u


def process_fidelity_unitary(source, target):
    return process_fidelity(Operator(source), target=Operator(target))


def process_fidelity_vec(source, target):
    return process_fidelity_unitary(
        u.vec_to_unitary(source),
        u.vec_to_unitary(target)
    )
