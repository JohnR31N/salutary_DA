"""Canonical constants and validation helpers for CMC ImageNet-100."""

from __future__ import annotations

from collections.abc import Iterable

from allthemix.data.datasets.registry import IMAGENET100, is_dataset

IMAGENET100_NAME = "imagenet100"
IMAGENET100_DIR_CANDIDATES = (
    IMAGENET100_NAME,
    "imagenet-100",
    "imagenet_100",
    "ImageNet100",
    "ImageNet-100",
)

# Released by the CMC authors in HobbitLong/CMC/imagenet100.txt. Preserve the
# source order here for provenance; torchvision ImageFolder assigns labels from
# the sorted synset ids, exposed separately below.
CMC_IMAGENET100_SYNSETS = (
    "n02869837",
    "n01749939",
    "n02488291",
    "n02107142",
    "n13037406",
    "n02091831",
    "n04517823",
    "n04589890",
    "n03062245",
    "n01773797",
    "n01735189",
    "n07831146",
    "n07753275",
    "n03085013",
    "n04485082",
    "n02105505",
    "n01983481",
    "n02788148",
    "n03530642",
    "n04435653",
    "n02086910",
    "n02859443",
    "n13040303",
    "n03594734",
    "n02085620",
    "n02099849",
    "n01558993",
    "n04493381",
    "n02109047",
    "n04111531",
    "n02877765",
    "n04429376",
    "n02009229",
    "n01978455",
    "n02106550",
    "n01820546",
    "n01692333",
    "n07714571",
    "n02974003",
    "n02114855",
    "n03785016",
    "n03764736",
    "n03775546",
    "n02087046",
    "n07836838",
    "n04099969",
    "n04592741",
    "n03891251",
    "n02701002",
    "n03379051",
    "n02259212",
    "n07715103",
    "n03947888",
    "n04026417",
    "n02326432",
    "n03637318",
    "n01980166",
    "n02113799",
    "n02086240",
    "n03903868",
    "n02483362",
    "n04127249",
    "n02089973",
    "n03017168",
    "n02093428",
    "n02804414",
    "n02396427",
    "n04418357",
    "n02172182",
    "n01729322",
    "n02113978",
    "n03787032",
    "n02089867",
    "n02119022",
    "n03777754",
    "n04238763",
    "n02231487",
    "n03032252",
    "n02138441",
    "n02104029",
    "n03837869",
    "n03494278",
    "n04136333",
    "n03794056",
    "n03492542",
    "n02018207",
    "n04067472",
    "n03930630",
    "n03584829",
    "n02123045",
    "n04229816",
    "n02100583",
    "n03642806",
    "n04336792",
    "n03259280",
    "n02116738",
    "n02108089",
    "n03424325",
    "n01855672",
    "n02090622",
)
CMC_IMAGENET100_CLASS_NAMES = tuple(
    sorted(
        CMC_IMAGENET100_SYNSETS,
    )
)
CMC_IMAGENET100_NUM_CLASSES = 100
CMC_IMAGENET100_TRAIN_EXAMPLES = 126_689
CMC_IMAGENET100_VAL_EXAMPLES = 5_000
CMC_IMAGENET100_SOURCE_URL = (
    "https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt"
)


def is_imagenet100_name(
    name: str,
) -> bool:
    """Return whether a dataset name is the canonical ImageNet-100 name."""
    return is_dataset(name, IMAGENET100)


def validate_cmc_imagenet100_class_names(
    class_names: Iterable[str],
    split_name: str,
) -> tuple[str, ...]:
    """Validate and return the fixed CMC classes in label-index order."""
    actual_class_names = tuple(
        sorted(
            class_names,
        )
    )
    actual_set = set(
        actual_class_names,
    )
    expected_set = set(
        CMC_IMAGENET100_CLASS_NAMES,
    )

    if len(actual_class_names) != len(actual_set):
        raise ValueError(
            f"CMC ImageNet-100 {split_name} contains duplicate class names."
        )

    if actual_set != expected_set:
        missing = sorted(
            expected_set - actual_set,
        )
        unexpected = sorted(
            actual_set - expected_set,
        )
        raise ValueError(
            f"CMC ImageNet-100 {split_name} class folders do not match the "
            "official CMC split. "
            f"Missing ({len(missing)}): {missing}; "
            f"unexpected ({len(unexpected)}): {unexpected}. "
            f"Official manifest: {CMC_IMAGENET100_SOURCE_URL}"
        )

    return CMC_IMAGENET100_CLASS_NAMES


def get_cmc_imagenet100_label_map() -> dict[str, int]:
    """Return the fixed torchvision-compatible synset-to-label mapping."""
    return {
        synset: label
        for label, synset in enumerate(
            CMC_IMAGENET100_CLASS_NAMES,
        )
    }


if len(CMC_IMAGENET100_SYNSETS) != CMC_IMAGENET100_NUM_CLASSES:
    raise RuntimeError("The bundled CMC ImageNet-100 manifest is incomplete.")

if len(set(CMC_IMAGENET100_SYNSETS)) != CMC_IMAGENET100_NUM_CLASSES:
    raise RuntimeError("The bundled CMC ImageNet-100 manifest has duplicates.")


__all__ = [
    "CMC_IMAGENET100_CLASS_NAMES",
    "CMC_IMAGENET100_NUM_CLASSES",
    "CMC_IMAGENET100_SOURCE_URL",
    "CMC_IMAGENET100_SYNSETS",
    "CMC_IMAGENET100_TRAIN_EXAMPLES",
    "CMC_IMAGENET100_VAL_EXAMPLES",
    "IMAGENET100_DIR_CANDIDATES",
    "IMAGENET100_NAME",
    "get_cmc_imagenet100_label_map",
    "is_imagenet100_name",
    "validate_cmc_imagenet100_class_names",
]
