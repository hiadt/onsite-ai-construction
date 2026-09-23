import pandas as pd

from demo.presentation import add_display_labels, attach_historical_labels, plain_rule_reasons, route_name, scene_name


def test_customer_labels_keep_distinct_routes_with_same_map_code():
    frame = pd.DataFrame({
        "sample_id": ["sample_b", "sample_a"],
        "route_id": ["map_023", "map_023"],
        "map_id": ["map_023", "map_023"],
    })
    shown = add_display_labels(frame, historical=True, frozen_ids=["sample_a", "sample_b"])
    assert shown["display_route"].tolist() == ["历史路线 002", "历史路线 001"]
    assert shown["display_scene"].tolist() == ["场景 23", "场景 23"]
    assert shown["sample_id"].tolist() == ["sample_b", "sample_a"]


def test_uploaded_filename_is_visible_without_exposing_hash():
    assert route_name("east-haul-v3.npz", 2, historical=False) == "候选路线 02 · east-haul-v3"
    assert scene_name("矿区东侧") == "矿区东侧"


def test_technical_rule_reason_has_plain_customer_copy():
    assert plain_rule_reasons("局部曲率或曲率变化相对开发集参考分布偏高") == "部分路段弯道较急或弯度变化较快"


def test_example_label_requires_matching_source_hash():
    examples = pd.DataFrame({"sample_id": ["a", "b"], "route_file_sha256": ["match", "changed"]})
    frozen = pd.DataFrame({"sample_id": ["a", "b"], "route_file_sha256": ["match", "old"], "true_label": [0, 1]})
    result = attach_historical_labels(examples, frozen)
    assert result.loc[0, "true_label"] == 0
    assert pd.isna(result.loc[1, "true_label"])
