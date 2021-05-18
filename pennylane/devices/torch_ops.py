#importing necessary libraries
import torch
from numpy import kron
from pennylane.utils import pauli_eigs

C_DTYPE = torch.complex128
R_DTYPE = torch.float64

#Another method to try
#I = torch.complex128([[1, 0], [0, 1]])
I = torch.tensor([[1, 0], [0, 1]], dtype=C_DTYPE, required_grad=True)
X = torch.tensor([[0, 1], [1, 0]], dtype=C_DTYPE, required_grad=True)
Y = torch.tensor([[0j, -1], [1j, 0j]], dtype=C_DTYPE, required_grad=True)
Z = torch.tensor([[1, 0], [0, -1]], dtype=C_DTYPE, required_grad=True)

#figure out if there is a method called torch.eye or other alternatives
II = torch.eye(4, dtype=C_DTYPE, required_grad=True)
ZZ = torch.tensor(kron(Z, Z), dtype=C_DTYPE, required_grad=True)

IX = torch.tensor((I, X), dtype=C_DTYPE, required_grad=True)
IY = torch.tensor((I, Y), dtype=C_DTYPE, required_grad=True)
IZ = torch.tensor((I, Z), dtype=C_DTYPE, required_grad=True)

ZI = torch.tensor(kron(Z, I), dtype=C_DTYPE, required_grad=True)
ZX = torch.tensor(kron(Z, X), dtype=C_DTYPE, required_grad=True)
ZY = torch.tensor(kron(Z, Y), dtype=C_DTYPE, required_grad=True)

def PhaseShift(phi):
    #Confirm the retun type syntax
    r"""One-qubit phase shift.

    Args:
        phi(float): phase shift angle

    Returns:
        torch.tensor[complex]: diagonal part of the phase shift matrix
    """
    #make sure if dtype of phi is changing else
    #replace it with phi.to(C_DTYPE)
    phi = torch.tensor(phi, dtype=C_DTYPE, required_grad=True)
    return torch.tensor([1.0, torch.exp(1j * phi)], required_grad=True)

def ControlledPhaseShift(phi):
    r"""Two-qubit controlled phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        torch.Tensor[complex]: diagonal part of the controlled phase shift matrix
    """
    phi = torch.tensor(phi, dtype=C_DTYPE, required_grad=True)
    return torch.tensor([1.0, 1.0, 1.0, torch.exp(1j * phi)], required_grad=True)


def RX(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        torch.tensor[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}
    """
    theta = torch.tensor(theta, dtype=C_DTYPE, required_grad=True)
    return torch.cos(theta / 2) * I + 1j * torch.sin(-theta / 2) * X

def RY(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle

    Returns:
        torch.tensor[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    theta = torch.tensor(theta, dtype=C_DTYPE, required_grad=True)
    return torch.cos(theta / 2) * I + 1j * torch.sin(-theta / 2) * Y

def RZ(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle

    Returns:
        torch.tensor[complex]: the diagonal part of the rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    theta = torch.tensor(theta, dtype=C_DTYPE, required_grad=True)
    p = torch.exp(-0.5j * theta)
    return torch.tensor([p, torch.conj(p)], required_grad=True)

#NOT WRITTEN
def Rot(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles

    Args:
        a, b, c (float): rotation angles

    Returns:
        torch.tensor[complex]: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    #Guess
    #return torch.diag(RZ(c)) @ RY(b) @ torch.diag(RZ(a))
    pass

def MultiRZ(theta, n):
    r"""Arbitrary multi Z rotation

    Args;
        theta (float): rotation angle
        n (int): number of wires the rotation acts on

    Returns:
        torch.tensor[complex]: diagonal part of the MultiRZ matrix
    """
    theta = torch.tensor(theta, dtype=C_DTYPE, required_grad=True)
    multi_Z_rot_eigs = torch.exp(-1j * theta / 2 * pauli_eigs(n))
    return torch.tensor(multi_Z_rot_eigs, required_grad=True)

def CRX(theta):
    r"""Two-quibit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        torch.tensor[complex]: unitary 4x4 rotation matrix :math: `|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle 1|\otimes
    """
    theta = torch.tensor(theta, dtype=C_DTYPE, required_grad=True)
    return(
        torch.cos(theta / 4) ** 2 * II
        - 1j * torch.sin(theta / 2) / 2 * IY
        + torch.sin(theta / 4) ** 2 * ZI
        + 1j * torch.sin(theta/ 2) / 2 * ZY
    )

def CRZ(theta):
    r"""Two-qubit controlled rotation about the z axis

    Args:
        theta (float): rotation angle

    Returns:
        torch.tensor[complex]: diagonal part of the 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    theta = torch.tensor(theta, dtype=C_DTYPE, required_grad=True)
    p = torch.exp(-0.5j * theta)
    return torch.tensor([1.0, 1.0, p, torch.conj(p)], required_grad=True)

#NOT WRITTEN
def CRot(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:

        array[complex]: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    #Guess
    #return torch.diag(CRZ(c)) @ (CRY(b) @ torch.diag(CRZ(a))
    pass

def SingleExcitation(phi):
    r"""Single excitation rotation

    Args:
        phi (float): rotataion angle

    Returns:
        torch.tensor[complex]: Single excitation rotation matrix
    """
    phi = torch.tensor(phi, dtype=C_DTYPE, required_grad=True)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)

    return torch.tensor([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], required_grad=True)

def SingleExcitationPlus(phi):
    r"""Single excitation rotation with positive phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle

    Returns:
        torch.tensor[complex]: Single excitation rotation matrix with positive phase-shift
    """
    phi = torch.cast(phi, dtype=C_DTYPE, required_grad=True)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(1j * phi / 2)
    return torch.tensor([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0][0, 0, 0, e]], required_grad=True)

def SingleExcitationMinus(phi):
    r"""Single excitation rotation with negative phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle
    
    Returns:
        torch.tensor[complex]: Single excitation rotation matrix with negative phase-shift
    """
    phi = torch.cast(phi, dtype=C_DTYPE, required_grad=True)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(-1j * phi / 2)
    return torch.tensor([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0][0, 0, 0, e]], required_grad=True)

def DoubleExcitation(phi):
    r"""Double excitation rotation.

    Args:
        phi (float): rotatation angle
    
    Returns:
        torch.tensor[complex]: Double excitatoin rotation matrix
    """
    phi = torch.cast(phi, dtype=C_DTYPE, required_grad=True)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    
    U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0, 0, -s, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, s, 0, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

    ]

    return torch.tensor(U, required_grad=True)

def DoubleExcitationPlus(phi):
    r"""double excitation rotation with positive phase-shift.

    Args:
        phi (float): rotation angle

    Returns:
        torch.tensor[complex]: rotation matrix
    """
    phi = torch.tensor(phi, dtype=C_DTYPE, required_grad=True)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(1j * phi / 2)

    U = [
        [e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0, 0, -s, 0, 0, 0],
        [0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0],
        [0, 0, 0, s, 0, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e],
    ]

    return torch.tensor(U), required_grad=True

def DoubleExcitationMinus(phi):
    r"""Double excitation rotation with negative phase with negative phase-shift.

    Args:
        phi (float): rotation angle

    Returns:
        torch.tensor[complex]: rotation matrix
    """
    phi = torch.tensor(phi, dtype=C_DTYPE, required_grad=True)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(-1j * phi / 2)

    U = [
        [e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0, 0, -s, 0, 0, 0],
        [0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0],
        [0, 0, 0, s, 0, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e],
    ]

    return torch.tensor(U, required_grad=True)








