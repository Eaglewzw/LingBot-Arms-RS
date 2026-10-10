import json
from pathlib import Path
import struct
import numpy as np
import pytest
import yaml
from rs_lingbot.bundle import inspect_checkpoint, prepare, robot_config, check_norm
from rs_lingbot.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    name = "model-00001-of-00001.safetensors"
    header = json.dumps({"weight": dict(dtype="F32", shape=[2], data_offsets=[0, 8])}).encode()
    (root / name).write_bytes(struct.pack("<Q", len(header)) + header + np.zeros(2, dtype="float32").tobytes())
    (root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"weight": name})))
    (root / "config.json").write_text("{}")
    return root


def test_partial_checkpoint_is_rejected(tmp_path):
    root = checkpoint(tmp_path)
    assert len(inspect_checkpoint(root)) == 1
    shard = next(root.glob("*.safetensors"))
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        inspect_checkpoint(root)


def test_index_mismatch_rejected(tmp_path):
    root = checkpoint(tmp_path)
    index = root / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"]["missing_tensor"] = data["weight_map"]["weight"]
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="index"):
        inspect_checkpoint(root)


def test_rs_bundle_has_upstream_compatible_layout_and_correct_single_mapping(tmp_path):
    root = checkpoint(tmp_path)
    training = tmp_path / "training.yaml"
    training.write_text(yaml.safe_dump(dict(model={}, train=dict(max_action_dim=55), data=dict(
        joints=[{"arm.position": 14}, {"end.position": 14}, {"effector.position": 2}],
        norm_type=[{"arm.position": "bounds_99_woclip"}, {"effector.position": "bounds_99_woclip"}],
        cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"]))))
    norm = tmp_path / "norm.json"
    stats = {}
    for prefix in ("observation.state", "action"):
        for feat, n in (("arm.position", 6), ("effector.position", 1)):
            stats[f"{prefix}.{feat}"] = {name: [value] * n for name, value in
                (("mean", 0), ("std", 1), ("q01", -1), ("q99", 1))}
    norm.write_text(json.dumps(dict(norm_stats=stats)))
    output = prepare(root, training, norm, ROOT / "configs/single_right.yaml", tmp_path / "bundle", rs_finetuned=True)
    manifest = json.loads((output / "manifest.json").read_text())
    path = output / manifest["checkpoint"]
    assert (path.parent.parent.parent / "lingbotvla_cli.yaml").is_file()
    assert next(path.glob("*.safetensors")).is_symlink()
    assert manifest["contract"]["rs_finetuned"]
    robot = yaml.safe_load((output / "robot.yaml").read_text())
    assert robot["actions"][0]["action.arm.position"]["origin_keys"] == [{"action": {"start": 0, "end": 6}}]
    assert robot["actions"][1]["action.effector.position"]["origin_keys"] == [{"action": {"start": 6, "end": 7}}]
    saved = yaml.safe_load((output / "lingbotvla_cli.yaml").read_text())
    assert isinstance(saved["data"]["joints"][0], str)


def test_robotwin_stats_cannot_be_mislabeled_as_single_rs(tmp_path):
    # RoboTwin uses 12 arm joints and 2 grippers. Keep this fixture local so
    # a fresh checkout can run tests without downloading LingBot or datasets.
    stats = {}
    for prefix in ("observation.state", "action"):
        for feature, dim in (("arm.position", 12), ("effector.position", 2)):
            stats[f"{prefix}.{feature}"] = {
                "mean": [0.] * dim, "std": [1.] * dim,
                "q01": [-1.] * dim, "q99": [1.] * dim,
            }
    norm = tmp_path / "robotwin_norm.json"
    norm.write_text(json.dumps({"norm_stats": stats}))
    check_norm(norm, 2)
    with pytest.raises(ValueError):
        check_norm(norm, 1)


def test_dual_mapping_is_left_then_right():
    c = load_config(ROOT / "configs/dual_stack_three.yaml")["contract_object"]
    robot = robot_config(c)
    arm = robot["actions"][0]["action.arm.position"]["origin_keys"]
    assert arm == [{"action": {"start": 0, "end": 6}}, {"action": {"start": 7, "end": 13}}]
