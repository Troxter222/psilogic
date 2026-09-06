# Security Policy

## Supported Versions

Security fixes are provided for the **latest release on PyPI** and the `main` branch.

| Version | Supported |
|---------|-----------|
| 0.6.x (latest) | Yes |
| 0.5.x | Best-effort until superseded |
| < 0.5 | No |

Always upgrade to the newest patch on PyPI when a security fix ships.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues privately using one of these channels:

1. **GitHub Security Advisories** (preferred): [Report a vulnerability](https://github.com/Troxter222/psilogic/security/advisories/new)
2. **Email**: troxtergrif@gmail.com — subject line `PsiLogic Security`

Include:

- A description of the issue and its potential impact
- Steps to reproduce (proof-of-concept if available)
- Affected version(s)
- Suggested fix, if you have one

We aim to acknowledge reports within **72 hours** and to publish a fix or mitigation plan within **14 days** for confirmed issues.

## Scope

### In scope

- The installable `psilogic` package (`psilogic/` on PyPI)
- State handling, checkpoint loading (`state_dict` migration), and distributed sync logic
- Optional integrations under `psilogic/integrations/` when used as documented

### Out of scope

- Third-party dependencies (PyTorch, HuggingFace, Lightning, etc.) — report those upstream
- Benchmark scripts under `benchmark/` (research tooling, not shipped on PyPI)
- Misconfiguration or misuse of training hyperparameters

## Security Design

The core optimizer is designed to be **offline and telemetry-free**:

| Property | Core package (`psilogic/`) |
|----------|----------------------------|
| Network calls | **None** — no HTTP, sockets, or remote logging |
| Telemetry / analytics | **None** |
| Arbitrary code execution | **None** — no `eval`, `exec`, or dynamic imports of user strings |
| Checkpoint loading | Uses PyTorch `state_dict` only; migration logic is version-tagged and explicit |

Optional benchmark tooling (`benchmark/`) may send notifications when `PSILOGIC_TG_TOKEN` and `PSILOGIC_TG_CHAT` environment variables are set. This code is **not** part of the PyPI wheel and is never invoked by `import psilogic`.

Framework integrations (`psilogic/integrations/`) delegate training to HuggingFace Trainer or PyTorch Lightning; network activity in those stacks is governed by the host framework, not PsiLogic.

## Disclosure

We follow coordinated disclosure. Credit will be given in the release notes unless you request anonymity.
