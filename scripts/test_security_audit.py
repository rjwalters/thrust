"""Exercise the exact audit entrypoint, including vulnerable graph controls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"RUSTSEC-2026-0367", "RUSTSEC-2026-0368", "RUSTSEC-2026-0369"}


class SecurityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/security-audit.sh", self.root / "scripts")
        shutil.copy2(ROOT / "deny.toml", self.root)
        (self.root / "lib.rs").write_text("pub fn audit_fixture() {}\n")

    def manifest(self, dependencies):
        (self.root / "Cargo.toml").write_text('''[package]
name = "thrust-audit-control"
version = "0.0.0"
edition = "2024"
publish = false
[workspace]
[lib]
path = "lib.rs"
''' + dependencies)

    def run_audit(self, env=None):
        return subprocess.run(["bash", "scripts/security-audit.sh"], cwd=self.root,
                              capture_output=True, text=True, timeout=180, env=env)

    def assert_vulnerable_graph(self, result):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        path = self.root / "target/security-audit"
        reports = [json.loads(line) for line in (path / "graph-audit.json").read_text().splitlines()]
        self.assertEqual(len(reports), 1)
        actual = {(item["advisory"]["id"], item["package"]["version"])
                  for item in reports[0]["vulnerabilities"]}
        expected = {(advisory, version) for advisory in EXPECTED
                    for version in ("0.6.6", "7.4.0")}
        self.assertTrue(expected <= actual, f"missing expected findings: {expected - actual}")
        self.assertEqual((path / "full-lock.exit").read_text().strip(), "1")
        self.assertTrue((path / "Cargo.lock").is_file())

    def test_optional_and_wasm_only_vulnerabilities_fail(self):
        self.manifest('''[features]
optional-backends = ["dep:zip06", "dep:zip74"]
[dependencies]
zip06 = { package = "zip", version = "=0.6.6", default-features = false, optional = true }
[target.'cfg(target_arch = "wasm32")'.dependencies]
zip74 = { package = "zip", version = "=7.4.0", default-features = false, optional = true }
''')
        self.assert_vulnerable_graph(self.run_audit())

    def test_build_and_dev_vulnerabilities_fail(self):
        self.manifest('''[build-dependencies]
zip06 = { package = "zip", version = "=0.6.6", default-features = false }
[dev-dependencies]
zip74 = { package = "zip", version = "=7.4.0", default-features = false }
''')
        (self.root / "build.rs").write_text("fn main() {}\n")
        self.assert_vulnerable_graph(self.run_audit())

    def fake_cargo(self, audit_body, audit_status=0, deny_status=0):
        directory = self.root / "bin"
        directory.mkdir()
        cargo = directory / "cargo"
        # Test responses are fixed JSON/text, never external input.
        cargo.write_text(f'''#!/usr/bin/env bash
case "$1" in
  generate-lockfile) printf 'version = 4\\n' > Cargo.lock ;;
  audit) cat <<'REPORT'
{audit_body}
REPORT
    exit {audit_status} ;;
  deny) printf 'simulated advisory database fetch failure\\n' >&2; exit {deny_status} ;;
  *) exit 99 ;;
esac
''')
        cargo.chmod(0o700)
        return {**os.environ, "PATH": str(directory) + os.pathsep + os.environ["PATH"]}

    def test_full_lock_fetch_failure_is_fatal(self):
        result = self.run_audit(self.fake_cargo("database fetch failed", audit_status=1))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "target/security-audit/graph-audit.exit").exists())

    def test_inconsistent_full_lock_exit_is_fatal(self):
        report = {"vulnerabilities": {"found": False, "count": 0, "list": []}}
        result = self.run_audit(self.fake_cargo(json.dumps(report), audit_status=1))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inconsistent report", result.stderr)

    def test_graph_database_fetch_failure_is_fatal(self):
        report = {"vulnerabilities": {"found": False, "count": 0, "list": []},
                  "database": {"last-commit": "fixture"},
                  "settings": {"ignore": [], "target_arch": [], "target_os": [], "severity": None}}
        result = self.run_audit(self.fake_cargo(json.dumps(report), deny_status=1))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simulated advisory database fetch failure", result.stdout)

    def test_missing_config_is_fatal(self):
        (self.root / "deny.toml").unlink()
        result = self.run_audit(self.fake_cargo("{}"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "Cargo.lock").exists())

    def test_invalid_config_is_fatal(self):
        self.manifest("")
        (self.root / "deny.toml").write_text("[invalid TOML")
        result = self.run_audit()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("config", result.stdout.lower() + result.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
