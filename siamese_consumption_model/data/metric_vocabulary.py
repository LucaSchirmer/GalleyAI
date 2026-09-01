"""Stable metric vocabulary shared by datasets, models, and reports."""

from __future__ import annotations


REGRESSION_METRICS = (
    "pct_bread_roll",
    "pct_brownie",
    "pct_chicken_rice_veg",
    "pct_fish_rice_veg",
    "pct_fruit_salad",
    "pct_rice",
    "pct_salad_dish_main",
    "pct_side_salad",
    "pct_vanilla_pudding",
    "pct_wrap_merged",
)

CLASSIFICATION_METRICS = (
    "drink_water",
    "drink_coffee",
    "drink_tea",
    "drink_oj",
    "drink_cola",
    "extra_butter",
    "extra_honey",
    "extra_plum_jam",
    "extra_cherry_jam",
    "status_cookie",
)

METRIC_NAMES = REGRESSION_METRICS + CLASSIFICATION_METRICS
METRIC_TO_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}


def metric_index(metric_name: str) -> int:
    try:
        return METRIC_TO_INDEX[metric_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Add it to data/metric_vocabulary.py "
            "before training so checkpoint semantics remain stable."
        ) from exc
3