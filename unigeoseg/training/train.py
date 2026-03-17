import logging
import math

import transformers
from transformers import HfArgumentParser, set_seed

from unigeoseg import conversation as conversation_lib
from unigeoseg.training.arguments import DataArguments, ModelArguments, UniGeoSegTrainingArguments
from unigeoseg.training.data import InstructionDataCollator, ResampledMultiTaskDataset, build_task_groups, summarize_task_groups
from unigeoseg.training.modeling import build_model, build_tokenizer, count_parameters
from unigeoseg.training.trainer import ProgressiveTaskCallback, UniGeoSegTrainer


LOGGER = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, UniGeoSegTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.mm_vision_tower is None and model_args.vision_tower is not None:
        model_args.mm_vision_tower = model_args.vision_tower

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()
    set_seed(training_args.seed)

    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_phi"]

    tokenizer = build_tokenizer(model_args)
    model, _ = build_model(model_args, training_args)
    total_params, trainable_params = count_parameters(model)
    LOGGER.info("Model parameters total=%.2fM trainable=%.2fM", total_params / 1e6, trainable_params / 1e6)

    task_groups = build_task_groups(data_args, tokenizer)
    task_sizes = summarize_task_groups(task_groups)
    if not task_sizes:
        raise ValueError("No training data was found. Please provide at least one dataset root or manifest.")
    LOGGER.info("Task-group sizes: %s", task_sizes)

    samples_per_epoch = data_args.samples_per_epoch or sum(task_sizes.values())
    train_dataset = ResampledMultiTaskDataset(
        task_groups=task_groups,
        samples_per_epoch=samples_per_epoch,
        seed=training_args.seed,
        pts_enabled=training_args.pts_enabled,
        interactive_weight_start=training_args.interactive_weight_start,
        interactive_weight_end=training_args.interactive_weight_end,
        referring_weight=training_args.referring_weight,
        reasoning_weight_start=training_args.reasoning_weight_start,
        reasoning_weight_end=training_args.reasoning_weight_end,
    )

    trainer = UniGeoSegTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=InstructionDataCollator(tokenizer),
        tokenizer=tokenizer,
        callbacks=[ProgressiveTaskCallback(train_dataset, total_epochs=math.ceil(training_args.num_train_epochs))],
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
