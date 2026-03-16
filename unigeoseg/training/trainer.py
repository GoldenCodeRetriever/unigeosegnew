import logging

from transformers import TrainerCallback


LOGGER = logging.getLogger(__name__)


class ProgressiveTaskCallback(TrainerCallback):
    def __init__(self, train_dataset, total_epochs: int):
        self.train_dataset = train_dataset
        self.total_epochs = int(max(total_epochs, 1))
        self.last_epoch = None

    def _apply_epoch(self, epoch_index: int):
        if self.last_epoch == epoch_index:
            return
        self.train_dataset.set_epoch(epoch_index, self.total_epochs)
        self.last_epoch = epoch_index
        LOGGER.info(
            "Resampled epoch %d with task weights=%s counts=%s",
            epoch_index + 1,
            self.train_dataset.current_weights,
            self.train_dataset.current_counts,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._apply_epoch(0)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._apply_epoch(int(state.epoch or 0))
