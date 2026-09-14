# Security

Patches go to the latest release on PyPI and to `main`.

| Version | Supported |
|---------|-----------|
| 0.6.x (latest) | yes |
| 0.5.x | until a newer line replaces it |
| < 0.5 | no |

When a fix ships, install the newest patch from PyPI.

## Reporting

Do not open a public GitHub issue for a vulnerability.

1. GitHub Security Advisories (preferred): [Report a vulnerability](https://github.com/Troxter222/psilogic/security/advisories/new)
2. Email [troxtergrif@gmail.com](mailto:troxtergrif@gmail.com), subject `PsiLogic Security`

Include what is wrong and what it can do, how to reproduce it (a proof-of-concept if you have one), which versions are affected, and a fix if you already have one.

I will reply within 72 hours. For a confirmed issue I try to ship a fix, or at least a mitigation note, within 14 days.

## Scope

In scope: the `psilogic` package on PyPI, checkpoint loading and `state_dict` migration, distributed sync, and `psilogic/integrations/` when used as documented.

Out of scope: PyTorch, HuggingFace, Lightning, and other dependencies (those go upstream); scripts under `benchmark/` (not in the wheel); a bad learning rate or other training settings.

## What the package does

`import psilogic` does not open a network connection and does not send telemetry. There is no `eval`, `exec`, or dynamic import of a user string. Checkpoints are PyTorch `state_dict` only. Migration is version-tagged and written out in code, not loaded as arbitrary objects beyond what PyTorch already does.

`benchmark/` can notify Telegram if `PSILOGIC_TG_TOKEN` and `PSILOGIC_TG_CHAT` are set. That code is not in the PyPI wheel and is not imported with the package.

`psilogic/integrations/` passes training to HuggingFace Trainer or PyTorch Lightning. If those stacks talk to the network, that is the framework, not PsiLogic.

Credit goes in the release notes unless you ask to stay anonymous.
