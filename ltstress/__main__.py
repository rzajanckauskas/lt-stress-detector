"""ltstress – Lithuanian stress detector. Usage: ltstress <command> [arguments]; ltstress <command> -h."""
import importlib
import sys

# command -> (module with main(argv), description); modules are imported only when the command runs
COMMANDS = {
    "check": ("ltstress.check", "check manifest stress marks against the recordings -> word verdicts (JSONL)"),
    "aggregate": ("ltstress.aggregate", "summarize check output by word form (JSON + TSV)"),
}


def _usage() -> str:
    lines = ["Usage: ltstress <command> [arguments]", "", "Commands:"]
    lines += [f"  {k:<15}{description}" for k, (_, description) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage(), file=sys.stderr)
        raise SystemExit(0 if argv else 2)
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n\n{_usage()}", file=sys.stderr)
        raise SystemExit(2)
    importlib.import_module(COMMANDS[command][0]).main(rest)


if __name__ == "__main__":
    main()
