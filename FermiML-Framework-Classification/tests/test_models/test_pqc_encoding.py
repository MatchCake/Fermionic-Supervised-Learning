import numpy as np
import pytest
import torch

from fermiml_framework_classification.models.pqc_encoding import PQCEncoding
from tests.configs import ATOL_MATRIX_COMPARISON, TEST_SEED


class TestPQCEncoding:
    @pytest.mark.parametrize(
        "n_qubits, n_features, expected",
        [
            (-1, 10, 10),
            (None, 10, 10),
            (-1, 15, 16),
            (-1, 2, 2),
            (-1, 1, 2),
            (4, 10, 4),
            (2, 20, 2),
            (16, 10, 10),
            (16, 5, 6),
        ],
    )
    def test_resolve_n_qubits(self, n_qubits, n_features, expected):
        assert PQCEncoding.resolve_n_qubits(n_qubits, n_features) == expected

    @pytest.mark.parametrize(
        "rule, n_features, expected",
        [
            ("full", 20, 20),
            ("half", 20, 10),
            ("quarter", 20, 6),
            ("full", 15, 16),
            ("half", 16, 8),
            ("quarter", 16, 4),
            ("half", 10, 6),
            ("quarter", 10, 4),
            ("quarter", 6, 2),
            ("quarter", 2, 2),
            ("half", 1, 2),
            ("eighth", 20, 4),
            ("eighth", 16, 2),
            ("eighth", 6, 2),
        ],
    )
    def test_resolve_n_qubits_rules(self, rule, n_features, expected):
        assert PQCEncoding.resolve_n_qubits(rule, n_features) == expected

    def test_full_rule_matches_negative_request(self):
        for n_features in range(1, 21):
            assert PQCEncoding.resolve_n_qubits("full", n_features) == PQCEncoding.resolve_n_qubits(-1, n_features)

    def test_resolve_n_qubits_rejects_unknown_rule(self):
        with pytest.raises(ValueError):
            PQCEncoding.resolve_n_qubits("third", 10)

    @pytest.mark.parametrize("rule, expected_depth", [("full", 1), ("half", 2), ("quarter", 4), ("eighth", 5)])
    def test_rules_set_the_depth(self, rule, expected_depth):
        assert PQCEncoding.from_features(20, rule, TEST_SEED).depth == expected_depth

    @pytest.mark.parametrize("n_qubits", [0, 1, 3, 7])
    def test_resolve_n_qubits_rejects_odd_or_too_small(self, n_qubits):
        with pytest.raises(ValueError):
            PQCEncoding.resolve_n_qubits(n_qubits, 10)

    def test_resolve_n_qubits_rejects_no_features(self):
        with pytest.raises(ValueError):
            PQCEncoding.resolve_n_qubits(-1, 0)

    @pytest.mark.parametrize(
        "n_features, n_qubits, expected_depth",
        [(10, -1, 1), (10, 2, 5), (10, 4, 3), (20, 8, 3), (20, 16, 2), (15, -1, 1), (15, 8, 2), (2, 2, 1), (3, 2, 2)],
    )
    def test_depth_follows_eq_8(self, n_features, n_qubits, expected_depth):
        assert PQCEncoding.from_features(n_features, n_qubits, TEST_SEED).depth == expected_depth

    def test_reference_angles_are_seeded_and_in_range(self):
        first = PQCEncoding.from_features(12, -1, TEST_SEED).reference_angles
        again = PQCEncoding.from_features(12, -1, TEST_SEED).reference_angles
        other = PQCEncoding.from_features(12, -1, TEST_SEED + 1).reference_angles
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)
        assert first.shape == (12,)
        assert len(set(first.tolist())) == 12
        assert np.all((first >= 0.0) & (first <= 1.0))

    def test_angles_follow_eq_7(self):
        encoding = PQCEncoding.from_features(3, 2, TEST_SEED)
        x = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]], dtype=torch.float64)
        expected = np.pi / 2 * encoding.reference_angles + np.pi / 2 * x.numpy()
        np.testing.assert_allclose(encoding.angles(x).numpy(), expected, atol=ATOL_MATRIX_COMPARISON)

    def test_angles_keep_dtype(self):
        encoding = PQCEncoding.from_features(3, 2, TEST_SEED)
        assert encoding.angles(torch.zeros(2, 3, dtype=torch.float32)).dtype == torch.float32

    def test_layers_load_every_feature_exactly_once(self):
        encoding = PQCEncoding.from_features(7, 4, TEST_SEED)
        angles = torch.arange(7.0).reshape(1, 7)
        loaded = torch.cat([encoding.layer_angles(angles, layer) for layer in range(encoding.depth)], dim=-1)
        np.testing.assert_array_equal(loaded.numpy(), angles.numpy())
        assert encoding.layer_angles(angles, 1).shape == (1, 3)

    @pytest.mark.parametrize(
        "n_qubits, layer, expected",
        [
            (4, 0, [(0, 1), (2, 3)]),
            (4, 1, [(1, 2)]),
            (4, 2, [(0, 1), (2, 3)]),
            (6, 1, [(1, 2), (3, 4)]),
            (2, 0, [(0, 1)]),
            (2, 1, []),
        ],
    )
    def test_entangler_pairs_alternate(self, n_qubits, layer, expected):
        encoding = PQCEncoding.from_features(n_qubits * 3, n_qubits, TEST_SEED)
        assert encoding.entangler_pairs(layer) == expected
