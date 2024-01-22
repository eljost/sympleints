import importlib

import numpy as np
import pytest
import tempfile

import sympleints.testing.intor as intor
import sympleints.testing.pyscf_interface as pi
import sympleints.main


def gen_and_compare_to_pyscf(key, name, ncomponents, pyscf_name, spherical, L_max=2):
    with tempfile.TemporaryDirectory(prefix="sympleints_test_") as out_dir:
        results = sympleints.main.run(
            l_max=L_max,
            l_aux_max=L_max,
            sph=spherical,
            keys=(key,),
            out_dir=out_dir,
            normalization=sympleints.main.Normalization.CGTO,
            boys_func="sympleints.testing.boys",
        )
        # Get path to generated integrals
        int_fn = results["python"][name]
        spec = importlib.util.spec_from_file_location(name, int_fn)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func_dict = getattr(module, name)

        # Compare against Helium-dimer w/ basis functions up to L_max
        mol = pi.get_he_chain(2, L_max=L_max)
        origin = np.full(3, 0.5)

        shells = pi.shells_from_pyscf_mol(mol)
        if key == "coul":
            integrals = list()
            for R, Z in zip(mol.atom_coords(), mol.atom_charges()):
                integrals.append(
                    -Z * intor.intor_2c(shells, func_dict, ncomponents, spherical, R=R)
                )
            integrals = np.sum(integrals, axis=0)
        else:
            integrals = intor.intor_2c(
                shells, func_dict, ncomponents, spherical, R=origin
            )

        # Compare against (renormalized) PySCF integrals
        with mol.with_common_orig(origin):
            integrals_ref = mol.intor(pyscf_name)
        # Bring our integrals in proper order for PySCF
        if spherical:
            Ls = [shell.L for shell in shells]
            P_sph = pi.get_sph_permutation_matrix(Ls)
            integrals = np.einsum(
                "ij,...jk,kl->...il", P_sph, integrals, P_sph.T, optimize="greedy"
            )
        # Ensure self-overlaps of 1.0 for Cartesian integrals
        else:
            integrals_ref *= pi.get_cart_norms(mol)

        np.testing.assert_allclose(integrals, integrals_ref, atol=1e-14)


@pytest.mark.parametrize("spherical", (False, True))
@pytest.mark.parametrize(
    "key, ncomponents, name, pyscf_name",
    (
        ("ovlp", 1, "ovlp3d", "int1e_ovlp_cart"),
        ("kin", 1, "kinetic3d", "int1e_kin_cart"),
        ("dpm", 3, "dipole3d", "int1e_r_cart"),
        # ("qpm", 6, "quadrupole3d", "int1e_rr_cart"),
        ("coul", 1, "coulomb3d", "int1e_nuc_cart"),
        ("2c2e", 1, "int2c2e3d", "int2c2e_cart"),
    ),
)
def test_integrals(key, name, ncomponents, pyscf_name, spherical):
    if spherical:
        pyscf_name = pyscf_name[:-4] + "sph"
    gen_and_compare_to_pyscf(key, name, ncomponents, pyscf_name, spherical, L_max=2)
