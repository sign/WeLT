from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from welt.model import WordLatentTransformer


class FreezeWarmupCallback(TrainerCallback):
    """
    If steps==0: no-op.
    Else: on train begin -> model.freeze_pretrained_models()
          after `steps` -> model.unfreeze() (once).
    Safe with DDP/Deepspeed since toggling requires_grad is fine mid-training.
    """

    def __init__(self, model: WordLatentTransformer, steps: int = 0):
        self.model = model

        self.steps = steps
        self.enabled = self.steps > 0

        print("FreezeWarmupCallback initialized with steps =", self.steps)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled:
            return control

        # If resuming past threshold, don't (re)freeze
        if state.global_step >= self.steps:
            self.enabled = False
            return control

        self.model.freeze_pretrained_models()
        print("✓ Freezing pretrained model parameters for first", self.steps, "steps.")

        control.should_log = True
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled:
            return control

        if state.global_step >= self.steps:
            self.model.unfreeze()
            print("✓ Unfreezing all model parameters after", state.global_step, "steps.")
            self.enabled = False

            # Force a log/save right after unfreeze for visibility/checkpoints
            control.should_log = True
            control.should_save = True
        return control
