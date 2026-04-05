from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prompt_choice() -> str:
    print("\nRestaurant Agent Test Runner")
    print("1) Run all tests")
    print("2) Run mock AI tests only")
    print("3) Run API mock test only")
    print("4) Custom pytest pattern")
    choice = input("Select an option [1-4]: ").strip() or "1"
    return choice


def build_pytest_command(choice: str) -> list[str]:
    base = [sys.executable, "-m", "pytest", "-vv", "-ra"]
    if choice == "2":
        return base + ["tests/test_ai_reasoner_mock.py"]
    if choice == "3":
        return base + ["tests/test_api_with_mock_ai.py"]
    if choice == "4":
        pattern = input("Enter pytest path or expression: ").strip()
        return base + ([pattern] if pattern else [])
    return base


def run_command(command: list[str]) -> int:
    print("\nRunning:")
    print(" ".join(command))
    print("\n--- Test Output ---")

    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    exit_code = process.wait()
    elapsed = time.perf_counter() - start
    print("\n--- Summary ---")
    print(f"Exit code: {exit_code}")
    print(f"Elapsed: {elapsed:.2f}s")
    return exit_code


def main() -> int:
    choice = prompt_choice()
    command = build_pytest_command(choice)
    if not command:
        print("No pytest pattern provided; nothing to run.")
        return 1
    return run_command(command)


if __name__ == "__main__":
    raise SystemExit(main())