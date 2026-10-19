# Temporary Lean environment setup

All downloaded tooling and third-party sources were placed under `/tmp`; the
canonical optimizer implementation was not edited.

1. `command -v lean lake elan` found no installed runtime.
2. A sandboxed `curl` to the official GitHub release host failed with
   `curl: (6) Could not resolve host: github.com`. The required retry with network
   escalation succeeded.
3. The official elan 4.2.4 archive was downloaded into `/tmp`, resumed after an
   initial 60-second timeout, and installed with `ELAN_HOME=/tmp/psilogic-elan`,
   `--no-modify-path --default-toolchain none`.
4. The official Lean 4.19.0 Linux archive was 343,842,845 bytes. A single stream
   was too slow and timed out after 1,200 seconds (62,208,581 bytes). A parallel
   download of 32 disjoint byte ranges completed. Each range's expected length
   was checked; concatenation and `tar --zstd -xf` succeeded.
5. Runtime output:
   ```text
   Lean (version 4.19.0, x86_64-unknown-linux-gnu, commit 6caaee842e94, Release)
   Lake version 5.0.0-6caaee8 (Lean version 4.19.0)
   ```
6. The official Mathlib v4.19.0 checkout resolved to
   `c44e0c8ee63ca166450922a373c7409c5d26b00b`. Its eight dependencies were fetched
   at the exact commits in `lake-manifest.json`.
7. `lake exe cache get` successfully built the 12-step cache executable, then
   attempted to write `/home/troxter/.cache/mathlib`; the filesystem sandbox
   denied that location. Subsequent commands use
   `MATHLIB_CACHE_DIR=/tmp/psilogic-mathlib-cache` and authorized network access.
8. The next cache attempt installed leantar 0.1.15 but could not resolve the
   ProofWidgets binary release because the shallow checkout had no tags:
   ```text
   error: no release tag found for revision 'c4919189477c3221e6a204008998b0d724f49904'
   ```
   Fetching official tags resolved that commit to `v0.0.57`; the cache download
   was retried with the six import paths used by `PsiLogic.lean`.

Final proof-check evidence is recorded separately in `build.log` and
`build_status.json`. This setup narrative is not a proof-check result.

9. After the ProofWidgets tag repair, the six-import cache request downloaded
   and unpacked **820 files, 100% success**, and exited 0.
10. The first complete proof-file check returned exit 1 because a real-valued
    definition was missing `noncomputable`; all ten theorem proof terms already
    elaborated. The exact failed output is retained in `build_attempt1.log`.
11. After adding `noncomputable` and cleaning unused-variable/style warnings,
    the final `lake env lean -DautoImplicit=false` check returned **exit 0 with
    no warnings**. All ten axiom reports list only `propext`, `Classical.choice`,
    and `Quot.sound`. See `build.log` and `build_status.json` for final evidence.
