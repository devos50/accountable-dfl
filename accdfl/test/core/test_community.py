from asyncio import sleep
from binascii import hexlify

import pytest

from accdfl.core.community import LearningCommunity
from accdfl.core.session_settings import LearningSettings, SessionSettings
from accdfl.test.util.fake_model_manager import FakeModelManager

from ipv8.test.base import TestBase
from ipv8.test.mocking.ipv8 import MockIPv8


class TestLearningCommunityBase(TestBase):
    NUM_NODES = 1
    TARGET_NUM_NODES = NUM_NODES
    DATASET = "cifar10"

    def create_node(self, *args, **kwargs):
        return MockIPv8("curve25519", self.overlay_class, *args, **kwargs)

    def setUp(self):
        super().setUp()

        self.initialize(LearningCommunity, self.NUM_NODES)

        learning_settings = LearningSettings(
            learning_rate=0.1,
            momentum=0.0,
            batch_size=1,
            weight_decay=0
        )

        self.settings = SessionSettings(
            dataset=self.DATASET,
            work_dir=self.temporary_directory(),
            learning=learning_settings,
            participants=[hexlify(node.my_peer.public_key.key_to_bin()).decode() for node in self.nodes],
        )

        for ind, node in enumerate(self.nodes):
            node.overlay.setup(self.settings)
            cur_model_mgr = node.overlay.model_manager
            node.overlay.model_manager = FakeModelManager(cur_model_mgr.model, self.settings,
                                                          cur_model_mgr.participant_index)


class TestLearningCommunity(TestLearningCommunityBase):

    @pytest.mark.asyncio
    async def test_availability_traces(self):
        self.nodes[0].overlay.start()
        assert self.nodes[0].overlay.is_active  # The node should be active at the start
        traces = {"active": [0.15], "inactive": [0]}
        self.nodes[0].overlay.set_traces(traces)
        await sleep(0.1)
        assert not self.nodes[0].overlay.is_active  # The node should be inactive at t=0.1
        await sleep(0.1)
        assert self.nodes[0].overlay.is_active  # The node should be active again at t=0.2
