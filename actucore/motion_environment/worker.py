"""A detached, durable environment task, independent of HTTP service lifecycle."""
import argparse

from .manager import read_json
from .recipe import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    execute(args.root, read_json(args.config), args.job_id)


if __name__ == "__main__":
    main()
