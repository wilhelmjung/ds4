#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"


def run_mu(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(MU), *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"mu {' '.join(args)} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def main() -> None:
    cpu = run_mu("--backend", "cpu", "--inspect")
    assert "mu backend=cpu" in cpu.stdout

    bad = run_mu("--backend", "bogus", "--inspect", check=False)
    assert bad.returncode == 2
    assert "--backend must be cpu or metal" in bad.stderr

    no_fallback_cpu = run_mu("--backend", "cpu", "--no-cpu-fallback", "--inspect")
    assert "mu backend=cpu" in no_fallback_cpu.stdout

    metal = run_mu("--backend", "metal", "--inspect", check=False)
    assert metal.returncode in (0, 1)
    if metal.returncode == 0:
        assert "mu backend=metal" in metal.stdout
    else:
        assert "Metal" in metal.stderr or "metal" in metal.stderr

    print("mu_backend_smoke ok")


if __name__ == "__main__":
    main()
