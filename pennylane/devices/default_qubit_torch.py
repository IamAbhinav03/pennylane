import numpy as np
from pennylane.operation import DiagonalOperation
from . import torch_ops
from . import DefaultQubit
from pytorch import torch

#code to check the pytorch version


#Made some changes, make it correct.
class DefaultQubitTorch(DefaultQubit):
    """Simulator plugin based on ``"default.qubit"``, written using PyTorch.

    **Short name:** ``default.qubit.torch``

    This device provides a pure-state qubit simulator written using PyTorch.
    As a result, it supports classical backpropagation as a means to compute the Jacobian. This can
    be faster than the parameter-shift rule for analytic quantum gradients
    when the number of parameters to be optimized is large.
    
    #Write the crct installation of Pytorch, current one is just dummy
    To use this device, you will need to install Pytorch:

    .. code-block:: console

        pip install pytoch 

    **Example**

    The ``default.qubit.torch`` is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) with the PyTorch interface. This is the default method
    of differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your TensorFlow computation:

    >>> dev = qml.device("default.qubit.tf", wires=1)
    >>> @qml.qnode(dev, interface="tf", diff_method="backprop")
    ... def circuit(x):
    ...     qml.RX(x[1], wires=0)
    ...     qml.Rot(x[0], x[1], x[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> weights = tf.Variable([0.2, 0.5, 0.1])
    >>> with tf.GradientTape() as tape:
    ...     res = circuit(weights)
    >>> print(tape.gradient(res, weights))
    tf.Tensor([-2.2526717e-01 -1.0086454e+00  1.3877788e-17], shape=(3,), dtype=float32)

    Autograph mode will also work when using classical backpropagation:

    >>> @tf.function
    ... def cost(weights):
    ...     return tf.reduce_sum(circuit(weights)**3) - 1
    >>> with tf.GradientTape() as tape:
    ...     res = cost(weights)
    >>> print(tape.gradient(res, weights))
    tf.Tensor([-3.5471588e-01 -1.5882589e+00  3.4694470e-17], shape=(3,), dtype=float32)

    There are a couple of things to keep in mind when using the ``"backprop"``
    differentiation method for QNodes:

    * You must use the ``"tf"`` interface for classical backpropagation, as TensorFlow is
      used as the device backend.

    * Only exact expectation values, variances, and probabilities are differentiable.
      When instantiating the device with ``analytic=False``, differentiating QNode
      outputs will result in ``None``.


    If you wish to use a different machine-learning interface, or prefer to calculate quantum
    gradients using the ``parameter-shift`` or ``finite-diff`` differentiation methods,
    consider using the ``default.qubit`` device instead.


    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.

        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means
            that the device returns analytical results.
            If ``shots > 0`` is used, the ``diff_method="backprop"``
            QNode differentiation method is not supported and it is recommended to consider
            switching device to ``default.qubit`` and using ``diff_method="parameter-shift"``.
    """

    name = "Default qubit (PyTorch) Pennylane plugin"
    short_name = "default.qubit.torch"
    pennylane_requires = '2'
    version = '0.0.1'
    author = 'Abhinav M Hari and Daniel Wang'

    parametric_ops = {
        "PhaseShift": torch_ops.PhaseShift,
        "ControlledPhaseShift": torch_ops.ControlledPhaseShift,
        "RX": torch.RX,
        "RY": torch.RY,
        "RZ": torch.RZ,
        "Rot": torch.Rot,
        "MultiRZ": torch.MultiRZ,
        "CRX": torch.CRX,
        "CRY": torch.CRY,
        "CRZ": torch.CRZ,
        "CRot": torch.CRot,
        "SingleExcitation": torch.SingleExcitation,
        "SingleExcitationPlus": torch.SingleExcitationPlus,
        "SingleExcitationMinus": torch.SingleExcitationMinus,
        "DoubleExcitation": torch.DoubleExcitation,
        "DoubleExcitationPlus": torch.DoubleExcitationPlus,
        "DoubleExcitationMinus": torch.DoubleExcitationMinus,
    
    }

    C_DTYPE = torch.complex128
    R_DTYPE = torch.float64
    #_asarray confusion, not sure
    _asarray = staticmethod(torch.tensor)
    _dot = staticmethod(lambda x, y: torch.tensordot(x, y, dim=1))
    _abs = staticmethod(torch.abs)
    #find alternative for tf.reduce_sum, or create one, 
    _reduce_sum = staticmethod()
    _reshape = staticmethod(torch.reshape)
    _flatten = staticmethod(lambda tensor: torch.reshape(tensor, [-1])) #not sure
    _gather = staticmethod(torch.gather)
    _einsum = staticmethod(torch.einsum)
    _cast = staticmethod(torch.tensor) #also check the torch.to function
    _transpose = staticmethod(torch.transpose)
    _tensordot = staticmethod(torch.tensordot)
    _conj = staticmethod(torch.conj)
    _imag = staticmethod(torch.imag)
    _roll = staticmethod(torch.roll) 
    _stack = staticmethod(torch.stack) #check if it is same as tf.stack)

    #maybe a extra static method for _asarray like in default_quibt_tf.py

    #special apply method
    @staticmethod
    def __init__(self, shots=1024, hardware_options=None):
        super().__init__(wires=24, shots=shots, analytic=False)
        self.hardware_options = hardware_options


    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            passthru_interface="torch",
            supports_reversible_diff=False
        )
        return capabilities

    @staticmethod
    #another static method for _scatter. Don't know what to do

    def _get_unitary_matrix(self, unitary):
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            torch.tensor[complex] or array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis, or in the case of a diagonal unitary,
            a 1D array representing the matrix diagonal. For non-parametric unitaries,
            the return type will be a ``np.ndarray``. For parametric unitaries, a ``torch.tensor``
            object will be returned.
        """
        op_name = unitary.name.split(".inv")[0]

        if op_name in self.parametric_ops:
            if op_name == "MultiRz":
                mat = self.parametric_ops[op_name](*unitary.parameters, len(unitary.wires))
            else:
                mat = self.parametric_ops[op_name](*unitary.parameters)

            if unitary.inverse:
                mat = self._transpose(self._conj(mat))

            return mat

        if isinstance(unitary, DiagonalOperation):
            return unitary.eigvals

        return unitary.matrix
