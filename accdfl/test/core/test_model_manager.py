import pytest

from accdfl.core.models import create_model
from accdfl.core.model_manager import ModelManager
from accdfl.core.session_settings import SessionSettings, LearningSettings


@pytest.fixture
def settings(tmpdir) -> SessionSettings:
    return SessionSettings(
        work_dir=str(tmpdir),
        dataset="cifar10",
        learning=LearningSettings(batch_size=20, learning_rate=0.002, momentum=0.9, weight_decay=0),
        participants=["a"],
    )


@pytest.fixture
def model_manager(tmpdir, settings):
    model = create_model(settings.dataset)
    return ModelManager(model, settings, 0)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_train(model_manager):
    await model_manager.train()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_compute_accuracy(model_manager):
    accuracy, loss = await model_manager.compute_accuracy(model_manager.model)
    assert accuracy > 0
    assert loss > 0
