# ZIP compatibility and security fork

The root Cargo patch replaces Candle 0.10.2, tch/torch-sys 0.22.0 and ZIP
9.0.5 with the exact source accepted during the October 2026 security audit.
It removes the consumers' old ZIP dependencies while preserving observable
archive semantics. The accepted 15-file patch has SHA256
`194c4da52459d51fd2d9d5518c59e435628eaba2a7531aef5732c8327e2bc328`.
`accepted.patch` records every change from the published crates.

`provenance.json` records each crates.io archive checksum, all published
baseline file hashes, every vendored file hash, and the test-support hashes.
The complete published source and MIT/Apache notices are retained. The
published consumer lockfiles are also retained for provenance; Cargo ignores
a dependency's own lockfile. Test-only resolved locks live in `test-locks/`.
The generated root `Cargo.lock` records this checkout's dependency resolution.

## Compatibility contract

ZIP's new `EntryArchive` preserves central-directory order and duplicates.
It shares the checked parser, content bounds, encryption and CRC handling
with the ordinary reader. The default `ZipArchive`/`Standard` behavior stays
unchanged. The declared Unicode-extra-field policy lets Candle retain its
strict policy while tch and torch-sys retain theirs. Candle preserves the
first name's position with the last duplicate's contents; tch retains every
entry in order. The consumers pin `=9.0.5` and call the new API, so removing
the ZIP override causes a compile error rather than silently changing semantics.

Candle disables ZIP's default features. tch and torch-sys enable
`aes-crypto,bzip2,deflate,time,zstd`. Features unify when both tensor libraries
are selected, so the harness runs the same fixtures separately
and in a package depending on both libraries, with an explicit combined-mode
oracle. **Combined mode extends Candle's accepted inputs:** deflate, bzip2
and zstd NPZ files become readable, whereas standalone Candle rejects them
without codec features. This is Cargo feature unification, not identical
behavior to the original two-version ZIP graph. All 16 combined Candle cases
were compared with unchanged published Candle 0.10.2 using ZIP 7.2.0 with the
same codec features; outputs and exit statuses agree. The manifest retains
both original standalone and measured original-source feature-enabled results.
No production source was changed to obtain this result.

## Reproduce the tests

Use Python 3.11+, Cargo, a C++ compiler, and native LibTorch **2.9**. The
[official CPU archive index](https://download.pytorch.org/libtorch/cpu) supplies
the Linux ZIP used in CI. Its pinned SHA256 is
`22de42abde933be46ce843467930bd0190b72271bfa2c11f84db95591a9834f1`, computed
from the official HTTPS download during integration (not a publisher signature).
The archive is a test dependency, not a production installation.

```sh
python3 security/zip-compat/run.py \
  --work-dir /tmp/thrust-zip-tests \
  --libtorch /path/to/libtorch \
  --extract-archive /path/to/libtorch.zip \
  --extract-sha256 22de42abde933be46ce843467930bd0190b72271bfa2c11f84db95591a9834f1
```

The work directory must be new and outside the checkout. On macOS, supply
the matching native PyTorch 2.9 wheel and its verified checksum, with
`--libtorch` pointing at its extracted `torch` directory. No Python package
installation is required. `--target-dir` optionally reuses a build cache.
Missing native dependencies fail the run; the LibTorch version check is not
bypassed. Logs, exit status and comparisons remain in the work directory.

The runner verifies source hashes, makes isolated copies, and enables
automatic test discovery only in the ZIP test copy. It runs the full upstream
ZIP library/integration suite plus EntryArchive regressions, the original
Candle NPZ/PTH and native tch serialization tests, all 31 accepted fixture
comparisons in each of two feature configurations, and the actual torch-sys
`extract()` function against the supplied official archive. Every extracted
file is compared with the archive, including the wheel's torch-to-libtorch
rename. ZIP test support comes from zip-rs/zip2 commit
`0123ed144cfa7d672dae78b79a034d8900f9aaeb`; the original upstream archive digest
is recorded in the provenance file.

The fixture manifest records outputs observed from the original consumers,
including duplicate, CP437, Unicode-extra-field, compression, ZIP64 and
Fortran-order cases. Its `combined` entries record the feature-enabled Candle
control described above, including the baseline binary digest. CI compares
those outputs without rebuilding or running the vulnerable baseline dependencies.

## Maintenance and retirement

On a ZIP bump, review and reapply the six ZIP source changes, update consumer
pins and provenance, and run the complete native harness and application
builds. Do not discard a compatibility fixture to make an upgrade pass.
Retire the fork when upstream provides equivalent entry-preserving and
declared-encoding APIs and both consumer migrations pass this corpus.
Upstream proposals are a separate outward-facing action; this integration
does not submit them.

This is source remediation. Optional packages in a lockfile are not evidence
that a deployed application uses them, and a successful source audit is not
runtime deployment evidence. Issue #356 tracks this checkout integration; no advisory warning is suppressed
by this patch. The library does not commit its generated root lockfile.

The root Cargo overrides apply only when this checkout is the build root.
Cargo does not propagate them into applications that depend on the published
`thrust-rl` crate. Vendored sources and this qualification corpus are excluded
from that package; downstream applications need their own equivalent overrides
until upstream migrations ship. The Tokio minimum of 1.53.3 is preserved in
the published dependency requirement.
