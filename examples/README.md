# Examples

Runnable recipes for integrating PsiLogic into common training stacks.

## HuggingFace Trainer — SST-2 Fine-Tuning

```bash
pip install psilogic[integrations] datasets
python examples/hf_sst2_finetune.py --max-steps 500
```

Uses `psilogic.integrations.hf.psilogic_trainer_class()` to swap the optimizer without changing the training loop.

## torchtune — Full Fine-Tune Config

`examples/torchtune/psilogic_full_finetune.yaml` is a reference YAML snippet showing how to wire PsiLogic into a torchtune recipe. Copy the optimizer block into your torchtune config and adjust `lr`, `gamma`, and `total_steps` for your run.

## Lightning

```python
from psilogic.integrations.lightning import configure_psilogic, ChaosMonitorCallback

trainer = configure_psilogic(model, lr=3e-4, total_steps=10_000)
trainer.callbacks.append(ChaosMonitorCallback(log_every_n_steps=100))
```

Install with `pip install psilogic[integrations]`.

## Auto-Configuration

For scripts that don't use a framework wrapper:

```python
from psilogic import PsiLogic

optimizer = PsiLogic.auto(model, total_steps=len(train_loader) * num_epochs)
```

Architecture (ViT / GPT / NLP encoder / CNN) is inferred from module types and parameter names.
