import torch
from torchvision.transforms import v2

NUM_CLASSES = 10

batch = {
    "imgs": torch.rand(4, 3, 224, 224),
    "classes": torch.randint(0, NUM_CLASSES, size=(4,)),
    "some_other_key": "this is going to be passed-through"
}

def labels_getter(batch):
    return batch["classes"]

out = v2.CutMix(num_classes=NUM_CLASSES, labels_getter=labels_getter)(batch)
print(f"{out['imgs'].shape = }, {out['classes'].shape = }, {out}")
