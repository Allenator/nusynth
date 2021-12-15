from abc import ABC, abstractmethod

from joblib import Parallel, delayed
import numpy as np
from tqdm.notebook import tqdm
from typing import Iterable

import nusynth.circuit as c
import nusynth.metric as m
import nusynth.squander as s
import nusynth.utils as u


class Ansatz(ABC):
    native_parallel = False

    inverse_cls = None
    fidelity_metric = lambda a, b: np.all(a == b).astype(float)


    def __init__(self, *args, **kwargs):
        self._init_inverse(*args, **kwargs)


    def __call__(self, input, **kwargs):
        return self.validate_forward(input, **kwargs)


    def __str__(self) -> str:
        ret = f'{self.__class__.__name__}: ' + \
            f'({self.dims_input}) → ({self.dims_output})'

        return ret


    def _init_inverse(self, *inverse_init_args, **inverse_init_kwargs):
        if self.inverse_cls is not None:
            self.inverse = self.inverse_cls(
                *inverse_init_args, **inverse_init_kwargs
            )


    @classmethod
    def get_fidelity(cls, source, target):
        return cls.fidelity_metric(source, target)


    def validate_fidelity(self, input, output, **kwargs):
        ret = self.inverse(output, **kwargs)

        return self.get_fidelity(input, ret)


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


    def _validate_dims_input(self, input):
        if len(input) != self.dims_input:
            raise ValueError(
                f'Input of length {len(input)} does not match '
                f'the specified length of {self.dims_input}'
            )


    def _validate_dims_output(self, output):
        if len(output) != self.dims_output: # type: ignore
            raise ValueError(
                f'Output of length {len(output)} does not match ' # type: ignore
                f'the specified length of {self.dims_output}'
            )


    def is_invalid_input(self, input):
        return np.any(np.isnan(input))


    def validate_forward(self, input, **kwargs):
        self._validate_dims_input(input)

        if self.is_invalid_input(input):
            err_ret = np.empty(self.dims_output)
            err_ret[:] = np.nan

            return err_ret

        output = self.forward(input, **kwargs)
        output = self.postprocess(input, output, **kwargs)

        self._validate_dims_output(output)

        return output


    def map(self, input_arr, **kwargs):
        with u.tqdm_joblib(tqdm(
                desc=f'{self.__str__()}', total=len(input_arr)
            )):
            output_arr = Parallel(n_jobs=self.parallel_n_jobs)(
                delayed(self.validate_forward)(i, **kwargs) for i in input_arr
            )

        return np.array(output_arr)


class Identity(Ansatz):
    def __init__(self, dims):
        self.dims = dims


    @property
    def dims_input(self):
        return self.dims


    @property
    def dims_output(self):
        return self.dims


    def forward(self, input, **kwargs):
        return input


class Layers(Ansatz):
    def __init__(self, ansatz_list):
        if not ansatz_list:
            return ValueError('List of ansaetze is empty')

        # dimensionality check
        do_prev = ansatz_list[0].dims_output
        for i_prev, a in enumerate(ansatz_list[1:]):
            di = a.dims_input
            do = a.dims_output
            if di != do_prev:
                a_prev = ansatz_list[i_prev]
                raise ValueError(
                    f'Mismatch between output dimension {do_prev} '
                    f'of {a_prev.__class__.__name__} (index {i_prev}) '
                    f'and input dimension {di} '
                    f'of {a.__class__.__name__} (index {i_prev + 1})'
                )
            do_prev = do

        self.ansatz_list = ansatz_list

        # TODO: automatically compute inverse for layered models
        super().__init__()


    def __getitem__(self, index):
        return self.ansatz_list[index]


    def _short_str(self):
        return f'{self.__class__.__name__} ({self.n_layers}): ' + \
            f'({self.dims_input}) → ({self.dims_output})'


    def __str__(self):
        ret = self._short_str()
        ret += '\n└ ' + '\n└ '.join(
            [f'{str(i)}: {a.__str__()}' for i, a in enumerate(self.ansatz_list)]
        )

        return ret


    @property
    def dims_input(self):
        return self.ansatz_list[0].dims_input


    @property
    def dims_output(self):
        return self.ansatz_list[-1].dims_output


    @property
    def n_layers(self):
        return len(self.ansatz_list)


    @property
    def native_parallel(self):
        for a in self.ansatz_list:
            if a.native_parallel:
                return True
        return False


    def forward(self, input, kwargs_dict={}):
        output = input
        for i, a in enumerate(self.ansatz_list):
            output = a.forward(output, **kwargs_dict.get(i, {}))

        return output


    def postprocess(self, input, output, kwargs_dict={}):
        return NotImplementedError(
            'Method "postprocess" cannot be directly called for layered models. '
            'Use "validate_forward" instead'
        )


    def validate_forward(self, input, kwargs_dict={}):
        output = input
        for i, a in enumerate(self.ansatz_list):
            output = a.validate_forward(output, **kwargs_dict.get(i, {}))

        return output


    def map_sequential(self, input_arr, kwargs_dict={}):
        with u.tqdm_joblib(tqdm(
                desc=f'{self._short_str()}', total=len(input_arr)
            )):
            output_arr = Parallel(n_jobs=self.parallel_n_jobs)(
                delayed(self.validate_forward)(i, kwargs_dict=kwargs_dict)
                for i in input_arr
            )

        return np.array(output_arr)


    def map_batch(self, input_arr, kwargs_dict={}):
        output_arr = input_arr
        for i, a in enumerate(tqdm(
            self.ansatz_list, desc=f'{self._short_str()}', unit='layer'
        )):
            output_arr = a.map(output_arr, **kwargs_dict.get(i, {}))

        return output_arr


    def map(self, input_arr, kwargs_dict={}):
        return self.map_batch(input_arr, kwargs_dict=kwargs_dict)


class CircuitAnsatz(Ansatz):
    def __init__(self, n_qubits, dims_input, circuit_constructor):
        self.n_qubits = n_qubits
        self._dims_input = dims_input
        self.circuit_constructor = circuit_constructor
        super().__init__()


    @property
    def dims_input(self):
        return self._dims_input


    @property
    def dims_output(self):
        return (2 ** self.n_qubits) ** 2 * 2


    def forward(self, input, **kwargs):
        qc = self.circuit_constructor(input, **kwargs)
        uni = u.circuit_to_unitary(qc)

        return u.unitary_to_vec(uni)


class XZRot_1Q(CircuitAnsatz):
    def __init__(self):
        super().__init__(1, 2, c.xz_rot_1q)


class U3Gate_3Q(CircuitAnsatz):
    def __init__(self):
        super().__init__(3, 9, c.u3_gate_3q)


class SU2_3Q(CircuitAnsatz):
    def __init__(self, n_reps):
        self.n_reps = n_reps
        super().__init__(
            3, 6 * (self.n_reps + 1),
            lambda input: c.su2_3q(input, self.n_reps)
        )


class QGAN_3Q(CircuitAnsatz):
    def __init__(self, n_reps):
        self.n_reps = n_reps
        super().__init__(
            3, 3 * (self.n_reps + 1),
            lambda input: c.qgan_3q(input, self.n_reps)
        )


class SquanderReconstruct(CircuitAnsatz):
    def __init__(self, n_qubits, n_layers_dict=s.default_n_layers_dict):
        self.n_layers_dict = s.default_n_layers_dict | n_layers_dict
        super().__init__(
            n_qubits, s.n_params(n_qubits, self.n_layers_dict),
            lambda input: c.squander(input, n_qubits, self.n_layers_dict)
        )


class SquanderDecompose(Ansatz):
    native_parallel=True

    inverse_cls = SquanderReconstruct
    fidelity_metric = m.process_fidelity_vec


    def __init__(
        self, n_qubits,
        n_layers_dict=s.default_n_layers_dict,
        initial_guess='zeros',
        tolerance=1e-12,
        fidelity_threshold=0.99,
        param_max=2 * np.pi,
        param_min=0,
        max_n_retries=10,
        verbose=False
    ):
        self.n_qubits = n_qubits
        self.n_layers_dict = s.default_n_layers_dict | n_layers_dict
        self.initial_guess = initial_guess
        self.tolerance = tolerance
        self.fidelity_threshold = fidelity_threshold
        self.param_max = param_max
        self.param_min = param_min
        self.max_n_retries = max_n_retries
        self.verbose = verbose

        s.assert_n_qubits(self.n_qubits)
        s.assert_n_layers_dict(self.n_qubits, self.n_layers_dict)

        super().__init__(self.n_qubits, self.n_layers_dict)


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
            or output.min() < self.param_min \
            or self.validate_fidelity(input, output) < self.fidelity_threshold:

            if n_retries < self.max_n_retries:
                kwargs['n_retries'] = n_retries + 1
                return self.validate_forward(input, **kwargs)
            else:
                err_ret = np.empty_like(output)
                err_ret[:] = np.nan
                return err_ret
        else:
            return output
