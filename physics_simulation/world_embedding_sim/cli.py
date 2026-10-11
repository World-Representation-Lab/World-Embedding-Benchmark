"""Command-line interface."""
import argparse
import json
from pathlib import Path

from .runner import run_example
from .queries import describe_output
from .config import validate_example


def main():
    """List examples, run a simulation, or describe existing results."""
    parser = argparse.ArgumentParser(prog="web-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List available examples")
    run = commands.add_parser("run", help="Simulate an example and render its video")
    run.add_argument("example", help="Configuration path or domain/family")
    run.add_argument("--runtime", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--cpus", type=int, default=2)
    describe = commands.add_parser(
        "describe", help="Generate query.json from an existing numerical run"
    )
    describe.add_argument("example", help="Configuration path or domain/family")
    describe.add_argument(
        "--output", required=True, type=Path, help="Existing example output directory"
    )
    args = parser.parse_args()
    examples = Path(__file__).parent / "examples"
    if args.command == "list":
        for path in sorted(examples.glob("*/*/case.json")):
            print(path.parent.relative_to(examples))
        return
    path = Path(args.example)
    if not path.is_file():
        path = examples / args.example / "case.json"
        if not path.resolve().is_relative_to(examples.resolve()):
            parser.error("Unknown example")
    if not path.is_file():
        parser.error(f"Example not found: {args.example}")
    config = json.loads(path.read_text())
    if args.command == "describe":
        validate_example(config)
        print(describe_output(config["domain"], args.output, config["family"]))
        return
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    run_example(
        config,
        json.loads(args.runtime.read_text()),
        args.output,
        args.cpus,
    )


if __name__ == "__main__":
    main()
