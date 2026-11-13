#!/usr/bin/env python3
"""Render checked site data directly into a portable skill definition."""
import argparse
import json
from pathlib import Path
from guide import effective, load_json, render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', type=Path, help='省略则仅检查；不写文件')
    a = p.parse_args()
    try:
        raw = load_json(a.config)
        result = render(raw)
        if a.output:
            a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'ok': True, **effective(raw)}, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as e:
        p.exit(1, f'失败：{e}\n')


if __name__ == '__main__':
    main()
