from abc import ABC, abstractmethod

from joblib import Parallel, delayed
import numpy as np
from tqdm.notebook import tqdm
from typing import Iterable

import nusynth.circuit as c
import nusynth.squander as s
import nusynth.utils as u


class Ansatz(ABC):
    def __call__(self, samples, **kwargs):
        return self.validate_forward(samples, **kwargs)


    native_parallel = False


    @property
    @abstractmethod
    def dims_input(self) -> int:
        ...


    @property
    @abstractmethod
    def dims_output(self) -> int:
        ...


    @property
    def parallel_n_jobs(self) -> int:
        if self.native_parallel:
            return 1
        else:
            return -1


    @abstractmethod
    def forward(self, input, **kwargs) -> Iterable:
        ...


    def postprocess(self, input, output, **kwargs) -> Iterable:
        return output


    def validate_forward(self, input, **kwargs):
        if len(input) != self.dims_input:
            raise ValueError(
                f'Input of length {len(input)} does not match '
                f'the specified length of {self.dims_input}'
            )

        output = self.forward(input, **kwargs)
        output = self.postprocess(input, output, **kwargs)

        if len(output) != self.dims_output: # type: ignore
            raise ValueError(
                f'Output of length {len(output)} does not match ' # type: ignore
                f'the specified length of {self.dims_output}'
            )

        return output


    def map(self, samples, **kwargs):
        with u.tqdm_joblib(tqdm(
                desc=f'{self.__class__.__name__}', total=len(samples)
            )):
            results = Parallel(n_jobs=self.parallel_n_jobs)(
                delayed(self.validate_forward)(i, **kwargs) for i in samples
            )

        return np.array(results)


class CircuitAnsatz(Ansatz):
    def __init__(self, dims_input, dims_output, circuit_constructor):
        self._dims_input = dims_input
        self._dims_output = dims_output
        self.circuit_constructor = circuit_constructor
        super().__init__()


    @property
    def dims_input(self):
        return self._dims_input


    @property
    def dims_output(self):
        return self._dims_output


    def forward(self, input, **kwargs):
        qc = self.circuit_constructor(input, **kwargs)
        uni = u.circuit_to_unitary(qc)

        return u.unitary_to_vec(uni)


class XZRot_1Q(CircuitAnsatz):
    def __init__(self):
        super().__init__(2, 8, c.xz_rot_1q)


class U3Gate_3Q(CircuitAnsatz):
    def __init__(self):
        super().__init__(9, 128, c.u3_gate_3q)


class SU2_3Q(CircuitAnsatz):
    def __init__(self, n_reps):
        self.n_reps = n_reps
        super().__init__(
            6 * (self.n_reps + 1), 128,
            lambda input: c.su2_3q(input, self.n_reps)
        )


class QGAN_3Q(CircuitAnsatz):
    def __init__(self, n_reps):
        self.n_reps = n_reps
        super().__init__(
            3 * (self.n_reps + 1), 128,
            lambda input: c.qgan_3q(input, self.n_reps)
        )


class Squander(Ansatz):
    def __init__(
        self, n_qubits,
        n_layers_dict=s.default_n_layers_dict,
        initial_guess='zeros',
        tolerance=1e-12,
        param_max=2 * np.pi,
        param_min=0,
        max_n_retries=10,
        verbose=False
    ):
        self.n_qubits = n_qubits
        self.n_layers_dict = s.default_n_layers_dict | n_layers_dict
        self.initial_guess = initial_guess
        self.tolerance = tolerance
        self.param_max = param_max
        self.param_min = param_min
        self.max_n_retries = max_n_retries
        self.verbose = verbose

        s.assert_n_qubits(self.n_qubits)
        s.assert_n_layers_dict(self.n_qubits, self.n_layers_dict)

        super().__init__()


    native_parallel=True


    @property
    def dims_input(self):
        return (2 ** self.n_qubits) ** 2 * 2


    @property
    def dims_output(self):
        return s.n_params(self.n_qubits, self.n_layers_dict)


    def forward(self, input, normalize=False, return_circuit=False, **kwargs):
        uni = u.vec_to_unitary(input)
        qc = s.decompose(
            uni,
            n_layers_dict=self.n_layers_dict,
            initial_guess=self.initial_guess,
            tolerance=self.tolerance,
            verbose=self.verbose
        )
        res = s.extract_params(qc, self.n_layers_dict, normalize=normalize)

        if return_circuit:
            return res, qc
        else:
            return res


    def postprocess(self, input, output, **kwargs):
        if 'n_retries' in kwargs:
            n_retries = kwargs['n_retries']
        else:
            n_retries = 0

        if np.any(np.isnan(output)) \
            or output.max() > self.param_max \
            or output.min() < self.param_min:

            if n_retries < self.max_n_retries:
                kwargs['n_retries'] = n_retries + 1
                return self.validate_forward(input, **kwargs)
            else:
                err_ret = np.empty_like(output)
                err_ret[:] = np.nan
                return err_ret
        else:
            return output
