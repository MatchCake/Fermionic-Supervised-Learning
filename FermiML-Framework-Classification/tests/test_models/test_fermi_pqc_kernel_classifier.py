import numpy as np
import pennylane as qml
import pytest
from matchcake.operations import CompRyRy, CompRzRz, CompZX
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from fermiml_framework_classification.experiments.single_experiment import (
    SingleExperiment,
)
from fermiml_framework_classification.models.fermi_pqc_kernel_classifier import (
    FermiPQCKernelClassifier,
)
from fermiml_framework_classification.models.pqc_encoding import PQCEncoding
from fermiml_framework_classification.models.pqc_kernel_classifier import (
    PQCKernelClassifier,
)
from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SINGLE_PRECISION_COMPARISON,
    TEST_SEED,
)


class TestFermiPQCKernelClassifier:
    @classmethod
    def _reference_state(cls, scaled_x: np.ndarray, encoding: PQCEncoding) -> np.ndarray:
        """Run the circuit of Fig. 1 on a statevector, gate by gate, with the full two-qubit matchgate matrices."""
        n_qubits = encoding.n_qubits
        angles = np.pi / 2 * encoding.reference_angles + np.pi / 2 * scaled_x
        # Built before the QNode: constructing an operation inside one would also queue it on the circuit.
        gates = []
        for layer in range(encoding.depth):
            layer_angles = angles[layer * n_qubits : (layer + 1) * n_qubits]
            if len(layer_angles) % 2:
                layer_angles = np.append(layer_angles, 0.0)
            for pair in range(len(layer_angles) // 2):
                wires = [2 * pair, 2 * pair + 1]
                params = np.array([layer_angles[2 * pair], layer_angles[2 * pair + 1]])
                gates.append((np.asarray(CompRyRy(params, wires=wires).matrix()), wires))
                gates.append((np.asarray(CompRzRz(params, wires=wires).matrix()), wires))
            for first in range(layer % 2, n_qubits - 1, 2):
                wires = [first, first + 1]
                gates.append((np.asarray(CompZX(wires=wires).matrix()), wires))

        @qml.qnode(qml.device("default.qubit", wires=n_qubits))
        def circuit():
            for matrix, wires in gates:
                qml.QubitUnitary(matrix, wires=wires)
            return qml.state()

        return np.asarray(circuit())

    @classmethod
    def _reference_kernel(cls, model: FermiPQCKernelClassifier, x0: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(x_train)
        states0 = np.stack([cls._reference_state(row, model.encoding_) for row in scaler.transform(x0)])
        states1 = np.stack([cls._reference_state(row, model.encoding_) for row in scaler.transform(x_train)])
        return np.abs(states0.conj() @ states1.T) ** 2

    @pytest.fixture
    def x_train(self):
        return np.random.default_rng(TEST_SEED).uniform(-3.0, 5.0, (14, 8)).astype(np.float32)

    @pytest.fixture
    def x_test(self):
        return np.random.default_rng(TEST_SEED + 1).uniform(-3.0, 5.0, (6, 8)).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1] * 7).astype(int)

    @pytest.fixture
    def fitted(self, x_train, y_train):
        return FermiPQCKernelClassifier(n_qubits=4, random_state=TEST_SEED).fit(x_train, y_train)

    def test_init(self):
        assert FermiPQCKernelClassifier().MODEL_NAME == "fPQC"

    def test_is_allowed_in_experiments(self):
        assert FermiPQCKernelClassifier in SingleExperiment.ALLOWED_MODEL_CLS

    @pytest.mark.parametrize(
        "n_qubits, n_features, expected", [(-1, 6, 6), (-1, 5, 6), (4, 6, 4), (2, 6, 2), (8, 6, 6)]
    )
    def test_wires_follow_the_encoding(self, n_qubits, n_features, expected):
        rng = np.random.default_rng(TEST_SEED)
        model = FermiPQCKernelClassifier(n_qubits=n_qubits).fit(rng.uniform(size=(8, n_features)), np.array([0, 1] * 4))
        assert model.n_qubits == expected

    @pytest.mark.parametrize("n_qubits", [4, 2, 8])
    def test_train_kernel_matches_statevector(self, n_qubits, x_train, y_train):
        model = FermiPQCKernelClassifier(n_qubits=n_qubits, random_state=TEST_SEED).fit(x_train, y_train)
        np.testing.assert_allclose(
            model.transform(x_train),
            self._reference_kernel(model, x_train, x_train),
            atol=ATOL_SINGLE_PRECISION_COMPARISON,
        )

    @pytest.mark.parametrize("n_features, n_qubits, depth", [(7, 4, 2), (11, 4, 3), (13, 4, 4), (9, 2, 5)])
    def test_partial_last_layer_matches_statevector(self, n_features, n_qubits, depth, y_train):
        # From depth 3 on the padded angle acts on a populated block, so these would catch a wrong pad value.
        x_train = np.random.default_rng(TEST_SEED).uniform(-1.0, 1.0, (14, n_features)).astype(np.float32)
        model = FermiPQCKernelClassifier(n_qubits=n_qubits, random_state=TEST_SEED).fit(x_train, y_train)
        assert model.encoding_.depth == depth
        np.testing.assert_allclose(
            model.transform(x_train),
            self._reference_kernel(model, x_train, x_train),
            atol=ATOL_SINGLE_PRECISION_COMPARISON,
        )

    def test_rectangular_kernel_matches_statevector_entry_by_entry(self, fitted, x_train, x_test):
        kernel = fitted.transform(x_test)
        assert kernel.shape == (6, 14)
        np.testing.assert_allclose(
            kernel, self._reference_kernel(fitted, x_test, x_train), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )

    def test_rectangular_kernel_does_not_mark_test_points_as_training_points(self, fitted, x_test):
        leading_block = fitted.transform(x_test)[:6, :6]
        assert not np.any(np.isclose(np.diag(leading_block), 1.0, atol=ATOL_SINGLE_PRECISION_COMPARISON))
        assert not np.allclose(leading_block, leading_block.T, atol=ATOL_SINGLE_PRECISION_COMPARISON)

    def test_train_kernel_is_a_valid_kernel(self, fitted, x_train):
        kernel = np.asarray(fitted.transform(x_train), dtype=np.float64)
        np.testing.assert_allclose(kernel, kernel.T, atol=ATOL_SINGLE_PRECISION_COMPARISON)
        np.testing.assert_allclose(np.diag(kernel), 1.0, atol=ATOL_SINGLE_PRECISION_COMPARISON)
        assert np.linalg.eigvalsh((kernel + kernel.T) / 2).min() > -ATOL_SINGLE_PRECISION_COMPARISON

    def test_predict_scales_inputs_once(self, fitted, x_train, x_test, y_train):
        reference = SVC(kernel="precomputed", C=fitted.svm_c).fit(
            self._reference_kernel(fitted, x_train, x_train), y_train
        )
        np.testing.assert_array_equal(
            fitted.predict(x_test), reference.predict(self._reference_kernel(fitted, x_test, x_train))
        )

    def test_predict_on_training_points_uses_the_kernel_the_svm_was_trained_on(self, fitted, x_train):
        reference = fitted.svm.predict(self._reference_kernel(fitted, x_train, x_train))
        np.testing.assert_array_equal(fitted.predict(x_train), reference)

    def test_predict_proba(self, fitted, x_test):
        probabilities = fitted.predict_proba(x_test)
        assert probabilities.shape == (6, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(
            probabilities,
            fitted.svm.predict_proba(self._reference_kernel(fitted, x_test, fitted.x_train_)),
            atol=ATOL_APPROX_COMPARISON,
        )

    def test_refit_on_another_dimension_rebuilds_the_circuit(self, fitted):
        rng = np.random.default_rng(TEST_SEED)
        x_new = rng.uniform(size=(8, 6))
        fitted.fit(x_new, np.array([0, 1] * 4))
        assert fitted.n_qubits == 4
        assert fitted.encoding_.depth == 2
        np.testing.assert_allclose(
            fitted.transform(x_new), self._reference_kernel(fitted, x_new, x_new), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )

    def test_shares_its_layout_with_the_unrestricted_model(self, fitted, x_train, y_train):
        unrestricted = PQCKernelClassifier(n_qubits=4, random_state=TEST_SEED).fit(x_train, y_train)
        np.testing.assert_array_equal(fitted.encoding_.reference_angles, unrestricted.encoding_.reference_angles)
        assert fitted.encoding_.depth == unrestricted.encoding_.depth
        assert fitted.encoding_.n_qubits == unrestricted.encoding_.n_qubits

    @pytest.mark.parametrize("n_features", [8, 5, 2])
    def test_double_gives_every_feature_its_own_wire_pair(self, n_features):
        rng = np.random.default_rng(TEST_SEED)
        model = FermiPQCKernelClassifier(n_qubits=FermiPQCKernelClassifier.DOUBLE_RULE).fit(
            rng.uniform(size=(8, n_features)), np.array([0, 1] * 4)
        )
        assert model.n_qubits == 2 * n_features
        assert model.encoding_.depth == 1

    @pytest.mark.parametrize("n_features", [1, 3, 5])
    def test_double_matches_the_unrestricted_full_kernel_on_odd_inputs(self, n_features):
        rng = np.random.default_rng(TEST_SEED)
        x_train, x_test = rng.uniform(size=(10, n_features)), rng.uniform(size=(4, n_features))
        y_train = np.array([0, 1] * 5)
        doubled = FermiPQCKernelClassifier(n_qubits=FermiPQCKernelClassifier.DOUBLE_RULE, random_state=TEST_SEED)
        doubled.fit(x_train, y_train)
        unrestricted = PQCKernelClassifier(n_qubits="full", random_state=TEST_SEED).fit(x_train, y_train)
        np.testing.assert_allclose(
            doubled.transform(x_test), unrestricted.transform(x_test), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )

    def test_an_unknown_rule_lists_double(self, x_train, y_train):
        with pytest.raises(ValueError, match="double"):
            FermiPQCKernelClassifier(n_qubits="dobule").fit(x_train, y_train)

    def test_double_matches_the_unrestricted_full_kernel(self, x_train, y_train, x_test):
        doubled = FermiPQCKernelClassifier(n_qubits=FermiPQCKernelClassifier.DOUBLE_RULE, random_state=TEST_SEED)
        doubled.fit(x_train, y_train)
        unrestricted = PQCKernelClassifier(n_qubits="full", random_state=TEST_SEED).fit(x_train, y_train)
        np.testing.assert_array_equal(doubled.encoding_.reference_angles, unrestricted.encoding_.reference_angles)
        np.testing.assert_allclose(
            doubled.transform(x_test), unrestricted.transform(x_test), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )

    @pytest.mark.parametrize("column", [0, 1, 6, 7])
    def test_double_reads_every_feature(self, x_train, y_train, x_test, column):
        model = FermiPQCKernelClassifier(n_qubits=FermiPQCKernelClassifier.DOUBLE_RULE, random_state=TEST_SEED)
        model.fit(x_train, y_train)
        shifted = x_test.copy()
        shifted[:, column] += 2.0
        assert np.abs(model.transform(shifted) - model.transform(x_test)).max() > 1e-2

    def test_full_reads_only_the_even_features(self, x_train, y_train, x_test):
        # The pair matchgates start from |00>, so their odd-parity block, which carries every odd feature, stays empty.
        model = FermiPQCKernelClassifier(n_qubits="full", random_state=TEST_SEED).fit(x_train, y_train)
        shifted = x_test.copy()
        shifted[:, 1::2] += 2.0
        np.testing.assert_allclose(
            model.transform(shifted), model.transform(x_test), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )
