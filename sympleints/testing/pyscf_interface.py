import numpy as np

from pyscf import gto
import scipy as sp

from sympleints.testing.shells import Shell


def get_mol(atoms: str, gbs_basis: dict) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = atoms
    basis = {atom: gto.basis.parse(atom_bas) for atom, atom_bas in gbs_basis.items()}
    mol.basis = basis
    mol.build()
    return mol


_L_MAP = {
    0: "S",
    1: "P",
    2: "D",
    3: "F",
    4: "G",
    5: "H",
    6: "I",
}


def get_he_chain(length=2, L_max=4):
    atoms = "; ".join([f"He 0.0 0.0 {i:.1f}" for i in range(length)])
    cgtos = list()
    for L in range(L_max + 1):
        L_char = _L_MAP[L]
        cgtos.append(
            f"""He {L_char}
                1.0 1.0
                2.0 1.0"""
        )
    gbs_basis = {
        "He": "\n".join(cgtos),
    }
    return get_mol(atoms, gbs_basis)


def get_cart_norms(mol: gto.Mole) -> np.ndarray:
    S = mol.intor("int1e_ovlp_cart")
    N = 1 / np.diag(S) ** 0.5
    NN = N[:, None] * N[None, :]
    return NN


def shells_from_pyscf_mol(mol):
    shells = list()
    for bas_id in range(mol.nbas):
        L = mol.bas_angular(bas_id)
        center = mol.bas_coord(bas_id)
        coeffs = mol.bas_ctr_coeff(bas_id).flatten()
        exps = mol.bas_exp(bas_id)
        assert coeffs.size == exps.size, "General contractions are not yet supported."
        center_ind = mol.bas_atom(bas_id)
        atomic_num = mol.atom_charge(center_ind)
        shell = Shell(L, center, coeffs, exps, center_ind, atomic_num)
        shells.append(shell)
    return shells


PYSCF_SPH_PS = {
    0: [[1]],  # s
    1: [[1, 0, 0], [0, 0, 1], [0, 1, 0]],  # px py pz
    2: [
        [0, 0, 0, 0, 1],  # dxy
        [0, 0, 0, 1, 0],  # dyz
        [0, 0, 1, 0, 0],  # dz²
        [0, 1, 0, 0, 0],  # dxz
        [1, 0, 0, 0, 0],  # dx² - y²
    ],
    3: [
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ],
    4: [
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
}


def get_sph_permutation_matrix(Ls):
    return sp.linalg.block_diag(*[PYSCF_SPH_PS[L] for L in Ls])
