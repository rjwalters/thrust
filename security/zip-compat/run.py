#!/usr/bin/env python3
"""Exercise the shipped ZIP fork and its native consumers in isolated copies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FEATURES = "aes-crypto,bzip2,deflate,time,zstd"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sources():
    provenance = json.loads((HERE / "provenance.json").read_text())
    if digest(HERE / "accepted.patch") != provenance["accepted_patch_sha256"]:
        raise RuntimeError("accepted patch digest mismatch")
    for relative, expected in provenance["vendored"].items():
        if digest(REPO / "vendor" / relative) != expected:
            raise RuntimeError(f"vendored source differs from accepted patch: {relative}")
    for crate in provenance["crates"]:
        name = f"{crate['name']}-{crate['version']}"
        actual = {f"{name}/{p.relative_to(REPO / 'vendor' / name)}"
                  for p in (REPO / "vendor" / name).rglob("*") if p.is_file()}
        expected = {p for p in provenance["vendored"] if p.startswith(name + "/")}
        if actual != expected:
            raise RuntimeError(f"unexpected or missing files in vendored {name}")
    for relative, expected in provenance["test_support"].items():
        if digest(HERE / relative) != expected:
            raise RuntimeError(f"test support digest mismatch: {relative}")
    cases = json.loads((HERE / "fixtures/manifest.json").read_text())["cases"]
    if len(cases) != 31:
        raise RuntimeError("expected the complete 31-case acceptance corpus")
    for case in cases:
        if digest(HERE / "fixtures" / case["file"]) != case["sha256"]:
            raise RuntimeError(f"fixture digest mismatch: {case['file']}")
    return provenance, cases


def prepare(work, provenance):
    work.mkdir(parents=True, exist_ok=False)
    for crate in provenance["crates"]:
        name = f"{crate['name']}-{crate['version']}"
        shutil.copytree(REPO / "vendor" / name, work / "candidate" / name)
    for backend in ["candle", "tch"]:
        shutil.copytree(HERE / f"candidate-{backend}", work / "harness" / f"candidate-{backend}")
    for name in ["combined", "extractor"]:
        shutil.copytree(HERE / name, work / "harness" / name)
    for name in ["candle-core-0.10.2", "tch-0.22.0", "zip-9.0.5"]:
        dest = work / "tests" / name
        shutil.copytree(work / "candidate" / name, dest)
        shutil.copy2(HERE / "test-locks" / f"{name}.lock", dest / "Cargo.lock")
    zt = work / "tests/zip-9.0.5"
    shutil.copytree(HERE / "zip-test-support", zt, dirs_exist_ok=True)
    manifest = zt / "Cargo.toml"
    content = manifest.read_text()
    if content.count("autotests = false\n") != 1:
        raise RuntimeError("ZIP test-copy manifest changed; review test discovery")
    manifest.write_text(content.replace("autotests = false\n", "autotests = true\n"))


def verify_extraction(archive, destination):
    """Compare every regular file to the official archive, including wheel rename."""
    expected = {}
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith("torch/"):
                name = "libtorch/" + name[len("torch/"):]
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("qualification archive contains a noncanonical path")
            with source.open(info) as stream:
                expected[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    actual = {str(p.relative_to(destination)): digest(p)
              for p in destination.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in destination.rglob("*")) or actual != expected:
        raise RuntimeError("torch-sys extraction differs from archive content")
    return {"files": len(actual), "archive_sha256": digest(archive)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="new directory outside the repository")
    parser.add_argument("--libtorch", type=Path, required=True, help="native LibTorch 2.9 installation")
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--extract-archive", type=Path, required=True, help="official LibTorch ZIP or PyTorch wheel")
    parser.add_argument("--extract-sha256", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    libtorch = args.libtorch.resolve()
    archive = args.extract_archive.resolve()
    if work == REPO or REPO in work.parents:
        parser.error("--work-dir must be outside the repository")
    if not (libtorch / "include/torch/csrc/api/include/torch/torch.h").is_file():
        parser.error("native LibTorch headers are missing")
    if not (libtorch / "lib").is_dir() or args.jobs < 1:
        parser.error("native LibTorch libraries and positive --jobs are required")
    if digest(archive) != args.extract_sha256:
        parser.error("qualification archive SHA256 mismatch")
    provenance, cases = verify_sources()
    prepare(work, provenance)
    target = (args.target_dir or work / "target").resolve()
    env = os.environ.copy()
    for key in ["LIBTORCH_USE_PYTORCH", "LIBTORCH_BYPASS_VERSION_CHECK", "LIBTORCH_INCLUDE",
                "LIBTORCH_LIB", "LIBTORCH_STATIC", "TORCH_CUDA_VERSION"]:
        env.pop(key, None)
    env.update(LIBTORCH=str(libtorch), CARGO_TARGET_DIR=str(target),
               CARGO_BUILD_JOBS=str(args.jobs), CARGO_TERM_COLOR="never")
    loader = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    env[loader] = str(libtorch / "lib")
    records = []
    result = {"passed": False, "stages": records, "comparisons": []}

    def run(name, cwd, command):
        print(f"START {name}", flush=True)
        started = time.monotonic()
        with (work / f"{name}.log").open("wb") as log:
            proc = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        records.append({"name": name, "command": command, "exit_code": proc.returncode,
                        "seconds": round(time.monotonic() - started, 2)})
        print(f"END {name}: {proc.returncode}", flush=True)
        if proc.returncode:
            raise RuntimeError(f"{name} failed; see {work / (name + '.log')}")

    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("cargo is required")
    base = [cargo, "test", "--locked", "-j", str(args.jobs)]
    patch = ["--config", "patch.crates-io.zip.path=" + json.dumps(str(work / "candidate/zip-9.0.5"))]
    torch = ["--config", "patch.crates-io.torch-sys.path=" + json.dumps(str(work / "candidate/torch-sys-0.22.0"))]
    try:
        run("zip-tests", work / "tests/zip-9.0.5",
            base + ["--lib", "--tests", "--no-default-features", "--features", FEATURES])
        run("candle-original-tests", work / "tests/candle-core-0.10.2",
            base + ["--test", "serialization_tests", "--test", "pth_tests"] + patch)
        run("tch-original-tests", work / "tests/tch-0.22.0",
            base + ["--test", "serialization_tests"] + patch + torch)
        for name in ["candidate-candle", "candidate-tch", "combined", "extractor"]:
            run(f"build-{name}", work / "harness" / name,
                [cargo, "build", "--locked", "--bins", "-j", str(args.jobs)])
        for mode in ["separate", "combined"]:
            for case in cases:
                prefix = "zip-parity-v6" if mode == "separate" else "zip-parity-combined"
                binary = target / "debug" / f"{prefix}-{case['backend']}"
                proc = subprocess.run([str(binary), str(HERE / "fixtures" / case["file"])],
                                      env=env, capture_output=True, text=True, timeout=30)
                expected = case.get("combined", case) if mode == "combined" else case
                passed = proc.returncode == expected["exit_code"] and proc.stdout == expected["stdout"]
                result["comparisons"].append({"mode": mode, "backend": case["backend"],
                    "file": case["file"], "passed": passed, "exit_code": proc.returncode,
                    "stdout": proc.stdout, "stderr": proc.stderr})
                if not passed:
                    raise RuntimeError(f"{mode} {case['backend']} changed compatibility for {case['file']}")
        destination = work / "extracted"
        destination.mkdir()
        run("torch-sys-extraction", work,
            [str(target / "debug/zip-compat-extractor"), str(archive), str(destination)])
        result["extraction"] = verify_extraction(archive, destination)
        result["test_totals"] = {}
        for name in ["zip-tests", "candle-original-tests", "tch-original-tests"]:
            counts = re.findall(r"test result: ok\. (\d+) passed;", (work / f"{name}.log").read_text())
            result["test_totals"][name] = sum(map(int, counts))
            if not counts or not sum(map(int, counts)):
                raise RuntimeError(f"no executed tests recorded for {name}")
        result["passed"] = True
    finally:
        (work / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "comparisons": len(result["comparisons"]),
                      "tests": result["test_totals"], "extraction": result["extraction"]}), flush=True)


if __name__ == "__main__":
    main()
