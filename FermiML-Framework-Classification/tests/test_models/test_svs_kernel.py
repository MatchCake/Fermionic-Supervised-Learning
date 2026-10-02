import pytest

from fermiml_framework_classification.models.svs_kernel import SVSKernel


class TestSVSKernel:
    @pytest.fixture
    def model(self):
        return SVSKernel()

    def test_init(self, model):
        assert model.MODEL_NAME == "SVSKernel"
