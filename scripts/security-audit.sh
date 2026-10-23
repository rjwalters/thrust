#!/usr/bin/env bash
# Required supported-build audit, with the broader lockfile report retained.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -f deny.toml
mkdir -p target/security-audit
report_dir=target/security-audit

# The library does not commit Cargo.lock. Retain this run's complete resolution.
cargo generate-lockfile
cp Cargo.lock "$report_dir/Cargo.lock"

# The lock includes upstream optional backends that Thrust cannot activate.
# Keep their findings visible; issue #356 tracks the unresolved upstream ZIPs.
audit_status=0
cargo audit --json > "$report_dir/full-lock.json" || audit_status=$?
printf '%s\n' "$audit_status" > "$report_dir/full-lock.exit"
python3 - "$report_dir/full-lock.json" "$audit_status" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    report = json.load(stream)
status = int(sys.argv[2])
findings = report["vulnerabilities"]
count = findings["count"]
if (type(count) is not int or count < 0 or len(findings["list"]) != count
        or findings["found"] is not (count > 0)
        or status != (1 if count else 0)):
    raise SystemExit("cargo-audit failed or returned an inconsistent report")
settings = report["settings"]
if (settings["ignore"] or settings["target_arch"] or settings["target_os"]
        or settings["severity"] is not None or not report["database"]["last-commit"]):
    raise SystemExit("full-lock diagnostic was filtered or lacks its database revision")
print(f"Full-lock cargo-audit exit code: {status}; vulnerability findings: {count}")
if count:
    print("::warning title=Full-lock advisory findings::"
          f"cargo-audit exited {status} with {count} findings. "
          "See the retained report and https://github.com/rjwalters/thrust/issues/356. "
          "Inactive upstream dependencies are not claimed patched.")
PY

# Explicit config, all features/workspace members, and no target/exclude flags.
# Fetch/configuration errors are fatal. Only maintenance notices are warnings.
graph_status=0
cargo deny --workspace --all-features --locked --format json --config deny.toml \
    check advisories --audit-compatible-output -W unmaintained \
    > "$report_dir/graph-audit.json" 2> "$report_dir/graph-audit.log" || graph_status=$?
printf '%s\n' "$graph_status" > "$report_dir/graph-audit.exit"
cat "$report_dir/graph-audit.log"
printf 'Supported-build advisory audit exit code: %s\n' "$graph_status"
exit "$graph_status"
