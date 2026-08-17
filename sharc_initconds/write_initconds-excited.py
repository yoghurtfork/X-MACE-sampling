"""Write the input file consumed by SHARC's ``excite.py`` utility."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ewin-low", type=float, default=1.6, help="lower excitation-window bound in eV")
    parser.add_argument("--ewin-high", type=float, default=3.3, help="upper excitation-window bound in eV")
    parser.add_argument("--seed", default="1234", help="random seed, or ! for system time")
    parser.add_argument(
        "--specified-states",
        type=int,
        nargs="+",
        metavar="STATE",
        help="use SHARC option 2 to select these state indices directly",
    )
    args = parser.parse_args(argv)
    if args.specified_states is None and args.ewin_low >= args.ewin_high:
        parser.error("--ewin-low must be lower than --ewin-high")
    if args.specified_states is not None and any(state < 1 for state in args.specified_states):
        parser.error("--specified-states values must be positive")
    with open("excite_inp.txt", "w", encoding="utf-8") as output:
        if args.specified_states is None:
            output.write(f"\n\n\n{args.ewin_low} {args.ewin_high}\n\n{args.seed}\n\n")
        else:
            output.write(f"\n\n2\n-inf inf\n{' '.join(map(str, args.specified_states))}\n\n")
    print("excite_inp.txt written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
