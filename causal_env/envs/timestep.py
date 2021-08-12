from dataclasses import dataclass, field

from typing import Any, List
import torch
import torch.utils.data as trchdata


@dataclass
class Timestep:
    info: Any = None
    context: torch.Tensor = field(repr=False, default=None)
    treatments: torch.Tensor = None
    causal_ids: torch.LongTensor = None
    reward: float = None
    done: bool = False
    id: int = None

class TimestepDataset(trchdata.Dataset):

    def __init__(self, dataset: List[Timestep]) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, i):
        ts = self.dataset[i]
        return ts.context, \
                ts.treatments, \
                ts.causal_ids, \
                ts.reward

    def __len__(self):
        return len(self.dataset)


