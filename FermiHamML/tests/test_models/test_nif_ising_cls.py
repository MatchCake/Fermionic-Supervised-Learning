import pytest

from fermihamml.models.nif_ising_cls import NIFIsingCls


class TestNIFIsingCls:
    def test_initialization(self):
        model = NIFIsingCls(
            input_shape=(28, 28),
            output_shape=(10,),
            n_qubits=6,
        )
        assert model.input_shape == (28, 28)
        assert model.output_shape == (10,)
        assert model.n_qubits == 6

    def test_initialization_n_qubits_none(self):
        model = NIFIsingCls(
            input_shape=(28, 28),
            output_shape=(10,),
            n_qubits=None,
        )
        assert model.input_shape == (28, 28)
        assert model.output_shape == (10,)
        assert model.n_qubits == 28 * 28

    def test_add_model_specific_args(self):
        parser = NIFIsingCls.add_model_specific_args()
        args = parser.parse_args([])
        assert args.n_qubits == NIFIsingCls.DEFAULT_N_QUBITS
        assert args.encoder_output_activation == NIFIsingCls.DEFAULT_ENCODER_OUTPUT_ACTIVATION

    def test_extra_repr(self):
        model = NIFIsingCls(
            input_shape=(28, 28),
            output_shape=(10,),
            n_qubits=6,
            encoder_output_activation="Tanh",
        )
        extra_repr = model.extra_repr()
        assert "n_qubits=6" in extra_repr
        assert "encoder_output_activation=Tanh" in extra_repr
