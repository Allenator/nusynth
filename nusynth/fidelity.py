from typing import Optional

import numpy as np

import nusynth.ansatz as a
import nusynth.metric as m


class Fidelity:
    def __call__(self, input, kwargs_dict={}):
        return self.forward(input, kwargs_dict=kwargs_dict)


    def __init__(
        self, metric,
        ansatz_a: a.Ansatz, ansatz_b: Optional[a.Ansatz]=None,
        exception_whitelist=[]
    ):
        if ansatz_b is None:
            if ansatz_a.dims_input != ansatz_a.dims_output:
                raise ValueError(
                    'Input and output of the first ansatz must be of equal '
                    'dimensions when the second ansatz is unspecified'
                )
            ansatz_b = a.Identity(ansatz_a.dims_input)
        if ansatz_a.dims_input != ansatz_b.dims_input:
            raise ValueError(
                'Input dimensions mismatch: '
                f'({ansatz_a.dims_input}), ({ansatz_b.dims_input})'
            )
        if ansatz_a.dims_output != ansatz_b.dims_output:
            raise ValueError(
                'Output dimensions mismatch: '
                f'({ansatz_a.dims_output}), ({ansatz_b.dims_output})'
            )

        self.metric = metric
        self.ansatz_a = ansatz_a
        self.ansatz_b = ansatz_b
        self.exception_whitelist = exception_whitelist

        self.input_arr = np.zeros((0, self.dims_input))
        self.output_arr_a = np.zeros((0, self.dims_output))
        self.output_arr_b = np.zeros((0, self.dims_output))
        self.fidelity_arr = np.zeros((0,))


    @property
    def dims_input(self):
        return self.ansatz_a.dims_input


    @property
    def dims_output(self):
        return self.ansatz_a.dims_output


    def metric_safe(self, source, target):
        try:
            return self.metric(source, target)
        except tuple(self.exception_whitelist) as _:
            return 0.0


    def map_metric(self, output_arr_a, output_arr_b):
        return np.array([
            self.metric_safe(output_arr_a[i], output_arr_b[i])
            for i in range(len(output_arr_a))
        ])


    def forward(self, input, kwargs_dict={}):
        output_a = self.ansatz_a(input, **kwargs_dict.get('a', {}))
        output_b = self.ansatz_b(input, **kwargs_dict.get('b', {}))

        return self.metric_safe(output_a, output_b)


    def map(self, input_arr, kwargs_dict={}):
        self.input_arr = input_arr

        self.output_arr_a = self.ansatz_a.map(
            self.input_arr, **kwargs_dict.get('a', {})
        )
        self.output_arr_b = self.ansatz_b.map(
            self.input_arr, **kwargs_dict.get('b', {})
        )

        self.fidelity_arr = self.map_metric(
            self.output_arr_a,
            self.output_arr_b
        )

        return self.fidelity_arr


class ProcessFidelity(Fidelity):
    def __init__(
        self, ansatz_a: a.Ansatz, ansatz_b: Optional[a.Ansatz]=None,
        exception_whitelist=[np.linalg.LinAlgError]
    ):
        super().__init__(
            m.process_fidelity_vec,
            ansatz_a, ansatz_b=ansatz_b,
            exception_whitelist=exception_whitelist
        )
