import argparse
import json
from .config import load_config
from .runtime import run


def main():
    parser = argparse.ArgumentParser(description="B601-RS LingBot inference runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["mock", "shadow", "live"], default="mock")
    parser.add_argument("--execute", action="store_true", help="enable single-arm motors; shutdown disables torque")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--policy-url")
    parser.add_argument("--check", action="store_true", help="validate config without camera/CAN/model access")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check:
        print(json.dumps(cfg["contract_object"].to_dict(), indent=2))
        return
    try:
        output = run(cfg, mode=args.mode, execute=args.execute, duration=args.duration, policy_url=args.policy_url)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Stopped: {exc}\n")
    print(f"Session logs: {output}")


if __name__ == "__main__":
    main()
