import logging
import os
from asyncio import sleep, CancelledError
from typing import Optional, Tuple

import libtorrent as lt

import torch.nn as nn

from accdfl.core.model import ModelType, serialize_model
from accdfl.util.torrent_utils import create_torrent_file
from ipv8.taskmanager import TaskManager
from ipv8.util import succeed, fail


class TorrentDownloadManager(TaskManager):
    """
    This manager manages the libtorrent model seeding and downloading.
    """

    def __init__(self, data_dir: str, participant_index: int):
        super().__init__()
        self.data_dir = data_dir
        self.participant_index = participant_index
        self.logger = logging.getLogger(__name__)
        self.model_downloads = {}
        self.model_torrents = {}
        self.trackers = []

        settings = {
            "enable_upnp": False,
            "enable_dht": False,
            "enable_lsd": False,
            "enable_natpmp": False,
            "allow_multiple_connections_per_ip": True,
            "min_reconnect_time": 1,
        }
        self.session = lt.session(settings)
        #self.session.set_alert_mask(lt.alert.category_t.all_categories)

    def _task_process_alerts(self):
        for alert in self.session.pop_alerts():
            self.logger.debug("(alert) %s", alert)

    def start(self, listen_port: int) -> None:
        self.logger.info("Starting libtorrent session, listening on port %d", listen_port)
        self.session.listen_on(listen_port, listen_port + 5)
        self.register_task("process_alerts", self._task_process_alerts, interval=1)

    def get_torrent_info(self, participant_index: int, round: int, model_type: ModelType) -> Optional[bytes]:
        if (participant_index, round, model_type) in self.model_torrents:
            return self.model_torrents[(participant_index, round, model_type)]
        return None

    def is_seeding(self, round: int, model_type: ModelType) -> bool:
        if (self.participant_index, round, model_type) in self.model_downloads:
            download = self.model_downloads[(self.participant_index, round, model_type)]
            if download.status().state == 5:  # Seeding
                return True
        return False

    def is_downloading(self, participant_index: int, round: int, model_type: ModelType) -> bool:
        return (participant_index, round, model_type) in self.model_downloads

    async def seed(self, round: int, model_type: ModelType, model: nn.Module):
        """
        Start seeding a given model if it is not seeding already.
        """
        if self.is_seeding(round, model_type):
            return succeed(None)

        # Serialize the model and store it in the data directory.
        model_name = "%d_%d_%s" % (self.participant_index, round, "local" if model_type == ModelType.LOCAL else "aggregated")
        model_file_path = os.path.join(self.data_dir, model_name)
        with open(model_file_path, "wb") as model_file:
            model_file.write(serialize_model(model))

        # Create a torrent and start seeding the model
        bencoded_torrent = create_torrent_file(model_file_path, trackers=self.trackers)
        self.model_torrents[(self.participant_index, round, model_type)] = bencoded_torrent
        torrent = lt.bdecode(bencoded_torrent)
        torrent_info = lt.torrent_info(torrent)
        seed_torrent_info = {
            "ti": torrent_info,
            "save_path": self.data_dir
        }
        upload = self.session.add_torrent(seed_torrent_info)
        #upload.auto_managed(False)
        #upload.resume()
        self.model_downloads[(self.participant_index, round, model_type)] = upload
        for _ in range(100):
            await sleep(0.1)
            if upload.status().state == 5:
                self.logger.info("Torrent seeding!")
                upload.force_reannounce()
                return succeed(None)

        return fail(RuntimeError("Torrent not seeding after 10 seconds!"))

    async def download(self, participant_index: int, round: int, model_type: ModelType, bencoded_torrent: bytes,
                       other_peer_lt_address: Tuple):
        self.model_torrents[(participant_index, round, model_type)] = bencoded_torrent
        torrent_info = lt.torrent_info(lt.bdecode(bencoded_torrent))
        download_torrent_info = {
            "ti": torrent_info,
            "save_path": self.data_dir
        }

        download = self.session.add_torrent(download_torrent_info)
        download.auto_managed(False)
        download.resume()
        self.model_downloads[(participant_index, round, model_type)] = download
        try:
            await sleep(1)
        except CancelledError:
            self.logger.warning("Ignoring cancellation of download task")

        while True:
            s = download.status()
            state_str = ['queued', 'checking', 'downloading metadata',
                         'downloading', 'finished', 'seeding', 'allocating', 'checking fastresume']
            self.logger.debug('%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s' %
                              (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000,
                               s.num_peers, state_str[s.state]))

            if not list(torrent_info.trackers()):
                self.logger.debug("No trackers available - will try to connect to seeder manually")
                download.connect_peer(other_peer_lt_address)
            if s.state == 4 or s.state == 5:
                # The download seems to be finished - wait until we have the file in disk
                model_name = "%d_%d_%s" % (
                self.participant_index, round, "local" if model_type == ModelType.LOCAL else "aggregated")
                model_file_path = os.path.join(self.data_dir, model_name)
                attempt = 1
                while not os.path.exists(model_file_path):
                    if attempt == 20:
                        self.logger.error("Download file not found after 20 retries!")
                        return None
                    await sleep(0.1)

                with open(model_file_path, "rb") as model_file:
                    serialized_model = model_file.read()
                return participant_index, round, model_type, serialized_model
            try:
                await sleep(0.2)
            except CancelledError:
                pass

    def stop_download(self, participant_index: int, round: int, model_type: ModelType):
        if (participant_index, round, model_type) not in self.model_downloads:
            return

        self.logger.info("Stopping download of %s model of participant %d for round %d",
                         "local" if model_type == ModelType.LOCAL else "aggregated", participant_index, round)
        download = self.model_downloads[(participant_index, round, model_type)]
        self.session.remove_torrent(download)
        self.model_downloads.pop((participant_index, round, model_type), None)
