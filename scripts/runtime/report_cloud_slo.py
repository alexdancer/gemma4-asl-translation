from __future__ import annotations

import json

from src.telemetry_slo import summarize


def main() -> int:
    report = summarize()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
