"""Prepare an explicit deployment bundle without changing the LingBot repository."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import yaml
import numpy as np
from .config import load_config


def inspect_checkpoint(directory):
    directory = Path(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    if not files:
        raise ValueError("empty checkpoint index")
    keys = {}
    for name in files:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("invalid checkpoint shard name")
        path = directory / name
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"incomplete shard: {name}")
            header_size = struct.unpack("<Q", raw)[0]
            if not 2 <= header_size <= 100_000_000:
                raise ValueError(f"invalid safetensors header: {name}")
            header = json.loads(handle.read(header_size))
        end = 0
        for key, item in header.items():
            if key == "__metadata__":
                continue
            start, offset = item["data_offsets"]
            if start < 0 or offset < start:
                raise ValueError("invalid tensor offsets")
            end = max(end, offset)
            if key in keys:
                raise ValueError(f"duplicate tensor: {key}")
            keys[key] = name
        if path.stat().st_size != 8 + header_size + end:
            raise ValueError(f"truncated or oversized shard: {name}")
    if keys != index["weight_map"]:
        raise ValueError("safetensors index does not match shard headers")
    return files


def robot_config(contract, diagnostic=False):
    # Raw state/action layout is [six joint radians, gripper open fraction] per arm.
    count = 2 if diagnostic else len(contract.arms)
    result = {}
    for category, prefix in (("states", "observation.state"), ("actions", "action")):
        features = []
        for feature, slices in (("arm.position", [(i * 7, i * 7 + 6) for i in range(count)]),
                                ("effector.position", [(i * 7 + 6, i * 7 + 7) for i in range(count)])):
            details = dict(origin_keys=[{prefix: dict(start=start, end=end)} for start, end in slices])
            if category == "actions":
                details["subtract_state"] = False
            features.append({f"{prefix}.{feature}": details})
        result[category] = features
    result["images"] = [f"observation.images.{name}" for name in contract.cameras]
    result["norm_stats"] = "norm_stats.json"
    return result


def check_norm(path, arms):
    stats = json.loads(Path(path).read_text())["norm_stats"]
    for prefix in ("observation.state", "action"):
        for feature, dim in (("arm.position", 6 * arms), ("effector.position", arms)):
            values = stats[f"{prefix}.{feature}"]
            for name in ("mean", "std"):
                a = np.asarray(values[name])
                if a.ndim != 1 or a.shape != (dim,) or not np.all(np.isfinite(a)):
                    raise ValueError(f"normalization {prefix}.{feature}.{name} must have shape ({dim},)")
            for name, values_array in values.items():
                a = np.asarray(values_array)
                if a.shape != (dim,) or not np.all(np.isfinite(a)):
                    raise ValueError(f"invalid normalization statistic {prefix}.{feature}.{name}")


def prepare(checkpoint, training_config, norm_stats, config, output, *, diagnostic=False, rs_finetuned=False):
    cfg = load_config(config)
    contract = cfg["contract_object"]
    if diagnostic and rs_finetuned:
        raise ValueError("RoboTwin diagnostics cannot be declared RS-finetuned")
    shards = inspect_checkpoint(checkpoint)
    check_norm(norm_stats, 2 if diagnostic else len(contract.arms))
    training = yaml.safe_load(Path(training_config).read_text())
    for key in ("model", "data", "train"):
        if key not in training:
            raise ValueError(f"training configuration is missing {key}")
    if training["train"].get("max_action_dim") != 55:
        raise ValueError("expected LingBot 55D training configuration")
    # Upstream FeatureInfo/get_normalizer expect literal dict strings from CLI serialization.
    for key in ("joints", "norm_type"):
        training["data"][key] = [repr(v) if isinstance(v, dict) else v for v in training["data"][key]]
    if not set(contract.cameras).issubset(training["data"]["cameras"]):
        raise ValueError("checkpoint data config is missing required camera slots")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    ckpt = root / "checkpoints" / "deployment" / "hf_ckpt"
    ckpt.mkdir(parents=True)
    for source in Path(checkpoint).resolve().iterdir():
        if source.is_file() and source.name in shards:
            (ckpt / source.name).symlink_to(source)
        elif source.is_file() and source.suffix in (".json", ".txt", ".jinja"):
            shutil.copy2(source, ckpt / source.name)
    shutil.copy2(training_config, root / "training_config.original.yaml")
    (root / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(training, sort_keys=False))
    shutil.copy2(norm_stats, root / "norm_stats.json")
    (root / "robot.yaml").write_text(yaml.safe_dump(robot_config(contract, diagnostic), sort_keys=False))
    metadata = contract.to_dict()
    metadata["rs_finetuned"] = rs_finetuned
    manifest = dict(contract=metadata, model_id=str(Path(checkpoint).resolve()),
                    source="robotwin-diagnostic" if diagnostic else "rs",
                    checkpoint="checkpoints/deployment/hf_ckpt", training_config="lingbotvla_cli.yaml",
                    robot_config="robot.yaml", norm_stats="norm_stats.json")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def main():
    p = argparse.ArgumentParser(description="Build a local, symlinked LingBot deployment bundle")
    for name in ("checkpoint", "training-config", "norm-stats", "config", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--robotwin-diagnostic", action="store_true", help="read-only 18K input/output diagnostics, not RS transfer")
    p.add_argument("--rs-finetuned", action="store_true", help="declare these weights/statistics were trained with this exact RS contract")
    a = p.parse_args()
    print(prepare(a.checkpoint, a.training_config, a.norm_stats, a.config, a.output,
                  diagnostic=a.robotwin_diagnostic, rs_finetuned=a.rs_finetuned))


if __name__ == "__main__":
    main()
