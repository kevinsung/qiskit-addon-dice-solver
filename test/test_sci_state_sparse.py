import os
import struct

import numpy as np
import pytest
from qiskit_addon_dice_solver import SCIStateSparse
from qiskit_addon_dice_solver.dice_solver import _read_dice_outputs

NORB = 4
NELEC = (2, 2)

# A determinant set that is NOT a Cartesian product of alpha and beta strings,
# which is the situation a sparse state representation is meant for.
# Each entry is (amplitude, alpha string, beta string).
DETS = [
    (0.9, 0b0011, 0b0011),
    (-0.3, 0b0101, 0b0011),
    (0.2, 0b0011, 0b1001),
    (0.1, 0b0110, 0b0110),
    # An exactly-zero amplitude must still be kept as its own determinant.
    (0.0, 0b1100, 0b0101),
]


def _to_dice_det_str(alpha: int, beta: int, norb: int) -> str:
    alpha_bits = f"{alpha:0{norb}b}"[::-1]
    beta_bits = f"{beta:0{norb}b}"[::-1]
    det = ""
    for a, b in zip(alpha_bits, beta_bits):
        if a == "1" and b == "1":
            det += "2"
        elif a == "1":
            det += "a"
        elif b == "1":
            det += "b"
        else:
            det += "0"
    return det


def _write_fake_dice_outputs(dice_dir, n_roots: int) -> None:
    """Write the files that _read_dice_outputs expects Dice to produce."""
    for root in range(n_roots):
        dets_file = "dets.bin" if root == 0 else f"dets_{root}.bin"
        with open(os.path.join(dice_dir, dets_file), "wb") as f:
            f.write(len(DETS).to_bytes(length=4, byteorder="little"))
            f.write(NORB.to_bytes(length=4, byteorder="little"))
            for amp, alpha, beta in DETS:
                f.write(struct.pack("<d", amp * (root + 1)))
                f.write(_to_dice_det_str(alpha, beta, NORB).encode())
        with open(os.path.join(dice_dir, f"spin1RDM.{root}.{root}.txt"), "w") as f:
            f.write(f"{2 * NORB}\n0 0 1.0\n1 1 1.0\n")
    with open(os.path.join(dice_dir, "shci.e"), "wb") as f:
        f.write(struct.pack("d" * n_roots, *[-1.0 - root for root in range(n_roots)]))


@pytest.mark.parametrize("n_roots", [1, 2])
def test_read_dice_outputs_sparse_state(tmp_path, n_roots):
    _write_fake_dice_outputs(tmp_path, n_roots)

    _, sci_state, _ = _read_dice_outputs(
        tmp_path, NORB, NELEC, n_roots=n_roots, return_sparse_state=True
    )

    states = sci_state if n_roots > 1 else [sci_state]
    assert len(states) == n_roots
    for root, state in enumerate(states):
        assert isinstance(state, SCIStateSparse)
        assert state.norb == NORB
        assert state.nelec == NELEC
        # One entry per determinant; compare order-independently
        actual = sorted(
            zip(state.ci_strs_a.tolist(), state.ci_strs_b.tolist(), state.amplitudes.tolist())
        )
        expected = sorted((a, b, amp * (root + 1)) for amp, a, b in DETS)
        assert len(actual) == len(expected)
        for (a, b, amp), (a_exp, b_exp, amp_exp) in zip(actual, expected):
            assert (a, b) == (a_exp, b_exp)
            assert amp == pytest.approx(amp_exp)


def test_sci_state_sparse_save_load(tmp_path):
    state = SCIStateSparse(
        amplitudes=np.array([amp for amp, _, _ in DETS]),
        ci_strs_a=np.array([a for _, a, _ in DETS], dtype=np.uint64),
        ci_strs_b=np.array([b for _, _, b in DETS], dtype=np.uint64),
        norb=NORB,
        nelec=NELEC,
    )
    filename = tmp_path / "state.npz"
    state.save(filename)
    loaded = SCIStateSparse.load(filename)

    np.testing.assert_array_equal(loaded.amplitudes, state.amplitudes)
    np.testing.assert_array_equal(loaded.ci_strs_a, state.ci_strs_a)
    np.testing.assert_array_equal(loaded.ci_strs_b, state.ci_strs_b)
    assert loaded.norb == NORB
    assert type(loaded.norb) is int
    assert loaded.nelec == NELEC
    assert all(type(n) is int for n in loaded.nelec)
