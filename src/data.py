from dataclasses import dataclass

import torch.utils.data


@dataclass(frozen=True)
class Relation:
    pass


class RelationDataset(torch.utils.data.Dataset[Relation]):
    pass
