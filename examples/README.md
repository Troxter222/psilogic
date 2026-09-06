# Examples

Runnable recipes for integrating PsiLogic into common training stacks.

**Defaults note (v0.6+):** bare `PsiLogic(...)` leaves AGC / grad centralization off.
Helpers below (`PsiLogic.auto`, HF/Lightning presets) may still enable mild AGC/GC.

## HuggingFace Trainer — SST-2 Fine-Tuning

```bash
pip install "psilogic[integrations]" datasets
python examples/hf_sst2_finetune.py --max-steps 500
```

Uses `psilogic.integrations.hf.psilogic_trainer_class()` to swap the optimizer without
changing the training loop. Expect a few minutes on GPU; slower on CPU.

## torchtune — Full Fine-Tune Config

`examples/torchtune/psilogic_full_finetune.yaml` is a **fragment** (optimizer block), not a
full runnable torchtune recipe. Copy it into your config and set `lr`, `gamma`, and
`gamma_T_max` / `total_steps` for your run.

## Lightning

`configure_psilogic` returns a **PsiLogic optimizer** for
`LightningModule.configure_optimizers` — it is not a `Trainer`.

```python
import lightning as L
from psilogic.integrations.lightning import configure_psilogic, ChaosMonitorCallback


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def configure_optimizers(self):
        return configure_psilogic(
            self.model,
            preset="auto",
            lr=3e-4,
            total_steps=int(self.trainer.estimated_stepping_batches),
        )


trainer = L.Trainer(callbacks=[ChaosMonitorCallback(log_every_n_steps=100)])
# trainer.fit(LitModel(model), train_dataloader)
```

Install with `pip install "psilogic[integrations]"`.

## Auto-Configuration

For scripts that don't use a framework wrapper:

```python
from psilogic import PsiLogic

optimizer = PsiLogic.auto(model, total_steps=len(train_loader) * num_epochs)
```

Architecture (ViT / GPT / NLP encoder / CNN) is inferred from module types and parameter names.

To add a new example, see [CONTRIBUTING.md](../CONTRIBUTING.md).
