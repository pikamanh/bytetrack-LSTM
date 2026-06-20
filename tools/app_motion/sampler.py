import copy
import random
import numpy as np

from collections import defaultdict
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """
    PK Sampler cho MOTReIDMotionDataset

    Dataset sample:
    {
        "track_id": xxx,
        "window": [...]
    }

    Batch:
        P identities
        K instances / identity

    batch_size = P * K
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        num_instances=4
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances

        if batch_size % num_instances != 0:
            raise ValueError(
                "batch_size phải chia hết cho num_instances"
            )

        self.num_pids_per_batch = (
            batch_size // num_instances
        )

        # ---------------------------------
        # Build pid -> sample indices
        # ---------------------------------

        self.index_dic = defaultdict(list)

        for idx, sample in enumerate(dataset.samples):

            pid = sample["track_id"]

            self.index_dic[pid].append(idx)

        self.pids = list(self.index_dic.keys())

        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError(
                f"Số identity ({len(self.pids)}) "
                f"nhỏ hơn P={self.num_pids_per_batch}"
            )

        # estimate epoch length
        self.length = 0

        for pid in self.pids:

            num = len(self.index_dic[pid])

            if num < self.num_instances:
                num = self.num_instances

            self.length += (
                num - num % self.num_instances
            )

    def __iter__(self):

        batch_idxs_dict = defaultdict(list)

        # ---------------------------------
        # Chuẩn bị K samples mỗi PID
        # ---------------------------------

        for pid in self.pids:

            idxs = copy.deepcopy(
                self.index_dic[pid]
            )

            if len(idxs) < self.num_instances:

                idxs = np.random.choice(
                    idxs,
                    size=self.num_instances,
                    replace=True
                )

            random.shuffle(idxs)

            batch_idxs = []

            for idx in idxs:

                batch_idxs.append(idx)

                if len(batch_idxs) == self.num_instances:

                    batch_idxs_dict[pid].append(
                        batch_idxs
                    )

                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)

        final_idxs = []

        # ---------------------------------
        # Chọn P identities mỗi batch
        # ---------------------------------

        while (
            len(avai_pids)
            >= self.num_pids_per_batch
        ):

            selected_pids = random.sample(
                avai_pids,
                self.num_pids_per_batch
            )

            for pid in selected_pids:

                batch_idxs = (
                    batch_idxs_dict[pid].pop(0)
                )

                final_idxs.extend(batch_idxs)

                if len(batch_idxs_dict[pid]) == 0:

                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length