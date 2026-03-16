import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from unigeoseg.constants import ANSWER_TOKEN_INDEX, IGNORE_INDEX, REFER_TOKEN_INDEX
from unigeoseg.eval_and_test.eval_dataset.RS_val_dataset import (
    extract_intera_coords,
    preprocess_image,
    preprocess_llama2,
    preprocess_mask,
    preprocess_referring_instruction,
)
from unigeoseg.eval_and_test.refer import REFER


PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


def _normalize_task_name(task_name: str) -> str:
    task = task_name.strip().lower()
    mapping = {
        "interactive": "interactive",
        "intera": "interactive",
        "interactive_seg": "interactive",
        "point": "interactive",
        "box": "interactive",
        "referring": "referring",
        "refer": "referring",
        "ref": "referring",
        "refer_seg": "referring",
        "reasoning": "reasoning",
        "reason": "reasoning",
        "reason_seg": "reasoning",
    }
    if task not in mapping:
        raise ValueError(f"Unsupported task type: {task_name}")
    return mapping[task]


def _task_to_prompt_type(task_name: str) -> str:
    return {"interactive": "intera", "referring": "ref", "reasoning": "reason"}[_normalize_task_name(task_name)]


def _task_to_dataset_type(task_name: str) -> str:
    _normalize_task_name(task_name)
    return "reason_seg"


def _safe_max_length(tokenizer) -> int:
    max_length = getattr(tokenizer, "model_max_length", 2048)
    if max_length is None or max_length <= 0 or max_length > 8192:
        return 2048
    return int(max_length)


def _resolve_optional_root(root: Optional[str], fallback_root: Optional[str], suffixes: Sequence[str]) -> Optional[str]:
    candidates: List[Path] = []
    if root:
        candidates.append(Path(root))
    if fallback_root:
        base = Path(fallback_root)
        candidates.append(base)
        candidates.extend(base / suffix for suffix in suffixes)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else None


def _resolve_path(value: str, base_dirs: Sequence[Path]) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    for base_dir in base_dirs:
        candidate = base_dir / value
        if candidate.exists():
            return str(candidate)
    return str((base_dirs[0] / value).resolve())


def _load_image_tensor(image_path: str, image_size: int) -> torch.Tensor:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = preprocess_image(image, image_size)
    return (torch.tensor(processed) - PIXEL_MEAN) / PIXEL_STD


def _load_mask_tensor(mask_source: Any, image_size: int) -> torch.Tensor:
    if isinstance(mask_source, (list, tuple)):
        mask = np.asarray(mask_source, dtype=np.uint8)
    else:
        source = str(mask_source)
        suffix = Path(source).suffix.lower()
        if suffix == ".npy":
            mask = np.load(source)
        elif Path(source).exists():
            mask = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.asarray(mask_source)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {mask_source}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    return preprocess_mask(mask, image_size)


def _format_bbox_instruction(bbox: Sequence[float]) -> str:
    x0, y0, x1, y1 = bbox
    return f"Please segment the region corresponding to the box x0,y0=[{x0},{y0}], x1,y1=[{x1},{y1}]."


def _format_points_instruction(points: Sequence[Sequence[float]]) -> str:
    point_text = ", ".join(f"[{x},{y}]" for x, y in points)
    return f"Please segment the target that contains the following point(s): {point_text}."


def _build_prompt_fields(tokenizer, task_name: str, instruction: str, answer: Optional[str] = None, intera_coords: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    task = _normalize_task_name(task_name)
    if task == "interactive":
        prefix = "This is an image <image>, Please doing Interactive Segmentation according to the following instruction:"
        response = "Sure. It is <seg>."
    elif task == "referring":
        prefix = "This is an image <image>, Please doing Referring Segmentation according to the following instruction:"
        response = "Sure. It is <seg>."
    else:
        prefix = "This is an image <image>, Please doing Reasoning Segmentation according to the following instruction:"
        response = "Sure, It is <seg>. \n<answer>."

    sources = [[
        {"from": "human", "value": prefix + "\n<refer>"},
        {"from": "gpt", "value": response},
    ]]
    text_dict = preprocess_llama2(sources, tokenizer)
    input_ids = text_dict["input_ids"][0]
    labels = text_dict["labels"][0]

    refer_instruction = preprocess_referring_instruction(f" {instruction}", tokenizer)
    refer_embedding_indices = torch.zeros_like(input_ids)
    refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1
    answer_embedding_indices = torch.zeros_like(input_ids)
    token_answer_id = None
    if task == "reasoning":
        token_answer_id = preprocess_referring_instruction(answer or "", tokenizer)
        answer_embedding_indices[input_ids == ANSWER_TOKEN_INDEX] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "dataset_type": _task_to_dataset_type(task),
        "prompt_type": _task_to_prompt_type(task),
        "token_refer_id": refer_instruction,
        "token_answer_id": token_answer_id,
        "refer_embedding_indices": refer_embedding_indices,
        "answer_embedding_indices": answer_embedding_indices,
        "intera_coords": intera_coords if task == "interactive" else None,
    }


class FilteredIndexDataset(Dataset):
    def __init__(self, parent: Dataset, indices: Sequence[int], task_name: str):
        self.parent = parent
        self.indices = list(indices)
        self.task_name = task_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.parent[self.indices[index]]


class GeoSegManifestDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer, image_size: int = 512, image_root: Optional[str] = None, mask_root: Optional[str] = None):
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.records = self._load_records(self.manifest_path)
        self.image_root = Path(image_root) if image_root else self.manifest_path.parent
        self.mask_root = Path(mask_root) if mask_root else self.manifest_path.parent
        self.task_to_indices: Dict[str, List[int]] = {"interactive": [], "referring": [], "reasoning": []}
        for index, record in enumerate(self.records):
            self.task_to_indices[_normalize_task_name(record["task"])].append(index)

    @staticmethod
    def _load_records(manifest_path: Path) -> List[Dict[str, Any]]:
        if manifest_path.suffix.lower() == ".jsonl":
            with manifest_path.open("r", encoding="utf-8") as file:
                return [json.loads(line) for line in file if line.strip()]
        with manifest_path.open("r", encoding="utf-8") as file:
            content = json.load(file)
        if isinstance(content, dict) and "samples" in content:
            return content["samples"]
        if not isinstance(content, list):
            raise ValueError(f"Unsupported manifest structure in {manifest_path}")
        return content

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        task_name = _normalize_task_name(record["task"])
        image_path = _resolve_path(record["image"], [self.image_root, self.manifest_path.parent])
        mask_value = record.get("mask")
        if isinstance(mask_value, str):
            mask_tensor = _load_mask_tensor(_resolve_path(mask_value, [self.mask_root, self.manifest_path.parent]), self.image_size)
        else:
            mask_tensor = _load_mask_tensor(mask_value, self.image_size)
        image_tensor = _load_image_tensor(image_path, self.image_size)

        instruction = record.get("instruction") or record.get("question") or record.get("text") or record.get("prompt") or ""
        answer = record.get("answer") or record.get("reason") or record.get("response")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        intera_coords = None
        if task_name == "interactive":
            if record.get("bbox") is not None:
                intera_coords = torch.tensor([record["bbox"]], dtype=torch.float32)
                if not instruction:
                    instruction = _format_bbox_instruction(record["bbox"])
            elif record.get("points") is not None:
                intera_coords = torch.tensor(record["points"], dtype=torch.float32)
                if not instruction:
                    instruction = _format_points_instruction(record["points"])
            else:
                intera_coords = extract_intera_coords(instruction)

        sample = _build_prompt_fields(self.tokenizer, task_name, instruction, answer=answer, intera_coords=intera_coords)
        sample["image"] = image_tensor
        sample["mask"] = mask_tensor
        sample["image_name"] = Path(image_path).stem
        return sample


class RefSegRSTrainDataset(Dataset):
    def __init__(self, dataset_root: str, tokenizer, split: str = "train", image_size: int = 512):
        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        with (self.dataset_root / f"output_phrase_{split}.txt").open("r", encoding="utf-8") as file:
            lines = file.readlines()
        self.samples: List[Dict[str, str]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            image_id, phrase = line.split(maxsplit=1)
            self.samples.append(
                {
                    "image": str(self.dataset_root / "images" / f"{image_id}.tif"),
                    "mask": str(self.dataset_root / "masks" / f"{image_id}.tif"),
                    "instruction": phrase,
                    "image_name": image_id,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_info = self.samples[index]
        sample = _build_prompt_fields(self.tokenizer, "referring", sample_info["instruction"])
        sample["image"] = _load_image_tensor(sample_info["image"], self.image_size)
        sample["mask"] = _load_mask_tensor(sample_info["mask"], self.image_size)
        sample["image_name"] = sample_info["image_name"]
        return sample


class RRSISDTrainDataset(Dataset):
    def __init__(self, dataset_root: str, tokenizer, split: str = "train", image_size: int = 512):
        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.refer = REFER(str(self.dataset_root), "rrsisd", "unc")
        self.ref_ids = list(self.refer.getRefIds(split=split))
        self.image_paths = [
            str(self.dataset_root / "images" / "rrsisd" / "JPEGImages" / self.refer.Imgs[ref_id]["file_name"])
            for ref_id in self.ref_ids
        ]

    def __len__(self) -> int:
        return len(self.ref_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ref_id = self.ref_ids[index]
        image_path = self.image_paths[index]
        sentence = self.refer.Refs[ref_id]["sentences"][0]["raw"]
        mask = self.refer.getMask(ref_id)
        sample = _build_prompt_fields(self.tokenizer, "referring", sentence)
        sample["image"] = _load_image_tensor(image_path, self.image_size)
        sample["mask"] = _load_mask_tensor(mask, self.image_size)
        sample["image_name"] = Path(image_path).stem
        return sample


class EarthReasonTrainDataset(Dataset):
    def __init__(self, dataset_root: str, tokenizer, split: str = "train", image_size: int = 512):
        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.images = sorted(path for path in (self.dataset_root / split / "images").iterdir() if path.is_file())
        self.labels = sorted(path for path in (self.dataset_root / split / "labels").iterdir() if path.is_file())
        self.qas = sorted(path for path in (self.dataset_root / split / "QAs").iterdir() if path.is_file())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.images[index]
        label_path = self.labels[index]
        qa_path = self.qas[index]
        with qa_path.open("r", encoding="utf-8") as file:
            qa_data = json.load(file)
        question = qa_data["questions"][0]
        answers = qa_data.get("answer", [])
        answer = answers[0] if answers else "There is no target object in the image."
        sample = _build_prompt_fields(self.tokenizer, "reasoning", question, answer=answer)
        sample["image"] = _load_image_tensor(str(image_path), self.image_size)
        sample["mask"] = _load_mask_tensor(str(label_path), self.image_size)
        sample["image_name"] = image_path.stem
        return sample


class RSRSTrainDataset(Dataset):
    def __init__(self, dataset_root: str, tokenizer, split: str = "train", image_size: int = 512, task_filter: Optional[str] = None):
        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.task_filter = _normalize_task_name(task_filter) if task_filter else None
        with (self.dataset_root / f"{split}.json").open("r", encoding="utf-8") as file:
            raw_samples = json.load(file)
        self.samples: List[Dict[str, Any]] = []
        for sample in raw_samples:
            task_name = self._infer_task(sample["img"])
            if self.task_filter and task_name != self.task_filter:
                continue
            enriched = dict(sample)
            enriched["task_name"] = task_name
            self.samples.append(enriched)

    @staticmethod
    def _infer_task(image_relative_path: str) -> str:
        normalized = image_relative_path.replace("\\", "/").lower()
        if "/intera/" in normalized:
            return "interactive"
        if "/ref/" in normalized:
            return "referring"
        return "reasoning"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_info = self.samples[index]
        image_path = self.dataset_root / sample_info["img"]
        mask_path = self.dataset_root / sample_info["mask"]
        json_path = self.dataset_root / sample_info["json"]
        with json_path.open("r", encoding="utf-8") as file:
            qa_data = json.load(file)
        questions = qa_data.get("question") or qa_data.get("questions") or [""]
        reasons = qa_data.get("reason") or qa_data.get("answer") or [""]
        instruction = questions[0]
        answer = reasons[0] if reasons else ""
        task_name = sample_info["task_name"]
        intera_coords = extract_intera_coords(instruction) if task_name == "interactive" else None
        sample = _build_prompt_fields(self.tokenizer, task_name, instruction, answer=answer, intera_coords=intera_coords)
        sample["image"] = _load_image_tensor(str(image_path), self.image_size)
        sample["mask"] = _load_mask_tensor(str(mask_path), self.image_size)
        sample["image_name"] = image_path.stem
        return sample


class InstructionDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = _safe_max_length(tokenizer)

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        refer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
            [instance["refer_embedding_indices"] for instance in instances],
            batch_first=True,
            padding_value=0,
        )
        answer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
            [instance["answer_embedding_indices"] for instance in instances],
            batch_first=True,
            padding_value=0,
        )

        input_ids = input_ids[:, : self.max_length]
        labels = labels[:, : self.max_length]
        refer_embedding_indices = refer_embedding_indices[:, : self.max_length]
        answer_embedding_indices = answer_embedding_indices[:, : self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "images": torch.stack([instance["image"] for instance in instances]),
            "masks": torch.stack([instance["mask"] for instance in instances]),
            "dataset_type": [instance["dataset_type"] for instance in instances],
            "prompt_type": [instance["prompt_type"] for instance in instances],
            "intera_coords": [instance.get("intera_coords") for instance in instances],
            "token_refer_id": [instance.get("token_refer_id") for instance in instances],
            "token_answer_id": [instance.get("token_answer_id") for instance in instances],
            "refer_embedding_indices": refer_embedding_indices,
            "answer_embedding_indices": answer_embedding_indices,
            "image_name": [instance.get("image_name") for instance in instances],
        }


class ResampledMultiTaskDataset(Dataset):
    def __init__(
        self,
        task_groups: Dict[str, List[Dataset]],
        samples_per_epoch: int,
        seed: int = 42,
        pts_enabled: bool = True,
        interactive_weight_start: float = 1.0,
        interactive_weight_end: float = 0.7,
        referring_weight: float = 1.0,
        reasoning_weight_start: float = 0.0,
        reasoning_weight_end: float = 0.3,
    ):
        self.task_groups = {name: [dataset for dataset in datasets if len(dataset) > 0] for name, datasets in task_groups.items()}
        self.task_groups = {name: datasets for name, datasets in self.task_groups.items() if datasets}
        if not self.task_groups:
            raise ValueError("No training datasets were found.")
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = seed
        self.pts_enabled = pts_enabled
        self.interactive_weight_start = interactive_weight_start
        self.interactive_weight_end = interactive_weight_end
        self.referring_weight = referring_weight
        self.reasoning_weight_start = reasoning_weight_start
        self.reasoning_weight_end = reasoning_weight_end
        self.sample_specs: List[Sequence[int]] = []
        self.current_weights: Dict[str, float] = {}
        self.current_counts: Dict[str, int] = {}
        self.current_epoch = 0
        self.total_epochs = 1
        self.set_epoch(0, 1)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _raw_group_weights(self, epoch_index: int, total_epochs: int) -> Dict[str, float]:
        if not self.pts_enabled:
            return {"interactive": 1.0, "referring": 1.0, "reasoning": 1.0}
        fraction = 1.0 if total_epochs <= 1 else epoch_index / float(total_epochs - 1)
        interactive_weight = self.interactive_weight_start + fraction * (self.interactive_weight_end - self.interactive_weight_start)
        reasoning_weight = self.reasoning_weight_start + fraction * (self.reasoning_weight_end - self.reasoning_weight_start)
        return {
            "interactive": interactive_weight,
            "referring": self.referring_weight,
            "reasoning": reasoning_weight,
        }

    def set_epoch(self, epoch_index: int, total_epochs: int) -> None:
        self.current_epoch = int(epoch_index)
        self.total_epochs = int(total_epochs)
        raw_weights = self._raw_group_weights(self.current_epoch, self.total_epochs)
        available_groups = [name for name in ("interactive", "referring", "reasoning") if name in self.task_groups]
        group_weights = [max(raw_weights.get(name, 0.0), 0.0) for name in available_groups]
        if sum(group_weights) <= 0:
            group_weights = [1.0 for _ in available_groups]
        weight_sum = float(sum(group_weights))
        normalized_weights = [weight / weight_sum for weight in group_weights]
        self.current_weights = {name: weight for name, weight in zip(available_groups, normalized_weights)}

        rng = random.Random(self.seed + self.current_epoch)
        group_dataset_weights = {name: [len(dataset) for dataset in self.task_groups[name]] for name in available_groups}
        self.current_counts = {name: 0 for name in available_groups}
        self.sample_specs = []
        for _ in range(self.samples_per_epoch):
            group_name = rng.choices(available_groups, weights=normalized_weights, k=1)[0]
            datasets = self.task_groups[group_name]
            dataset_index = rng.choices(range(len(datasets)), weights=group_dataset_weights[group_name], k=1)[0]
            sample_index = rng.randrange(len(datasets[dataset_index]))
            self.sample_specs.append((group_name, dataset_index, sample_index))
            self.current_counts[group_name] += 1

    def __getitem__(self, index: int) -> Dict[str, Any]:
        group_name, dataset_index, sample_index = self.sample_specs[index]
        return self.task_groups[group_name][dataset_index][sample_index]


def build_task_groups(data_args, tokenizer) -> Dict[str, List[Dataset]]:
    groups: Dict[str, List[Dataset]] = {"interactive": [], "referring": [], "reasoning": []}

    if data_args.geoseg_manifest:
        manifest_paths = [item.strip() for item in data_args.geoseg_manifest.split(",") if item.strip()]
        for manifest_path in manifest_paths:
            manifest_dataset = GeoSegManifestDataset(
                manifest_path,
                tokenizer,
                image_size=data_args.image_size,
                image_root=data_args.manifest_image_root,
                mask_root=data_args.manifest_mask_root,
            )
            for task_name, indices in manifest_dataset.task_to_indices.items():
                if indices:
                    groups[task_name].append(FilteredIndexDataset(manifest_dataset, indices, task_name))

    rrsisd_root = _resolve_optional_root(data_args.rrsisd_root, data_args.data_root, ["rs_ref_seg/RRSIS-D"])
    if rrsisd_root and Path(rrsisd_root).exists():
        groups["referring"].append(
            RRSISDTrainDataset(rrsisd_root, tokenizer, split=data_args.rrsisd_split, image_size=data_args.image_size)
        )

    refsegrs_root = _resolve_optional_root(data_args.refsegrs_root, data_args.data_root, ["rs_ref_seg/RefSegRS"])
    if refsegrs_root and Path(refsegrs_root).exists():
        groups["referring"].append(
            RefSegRSTrainDataset(refsegrs_root, tokenizer, split=data_args.refsegrs_split, image_size=data_args.image_size)
        )

    earthreason_root = _resolve_optional_root(data_args.earthreason_root, data_args.data_root, ["rs_reason_seg/RSReasonSeg", "RSReasonSeg"])
    if earthreason_root and Path(earthreason_root).exists():
        groups["reasoning"].append(
            EarthReasonTrainDataset(earthreason_root, tokenizer, split=data_args.earthreason_split, image_size=data_args.image_size)
        )

    if data_args.rsrs_root and Path(data_args.rsrs_root).exists():
        for task_name in ("interactive", "referring", "reasoning"):
            dataset = RSRSTrainDataset(
                data_args.rsrs_root,
                tokenizer,
                split=data_args.rsrs_split,
                image_size=data_args.image_size,
                task_filter=task_name,
            )
            if len(dataset) > 0:
                groups[task_name].append(dataset)

    return groups


def summarize_task_groups(task_groups: Dict[str, List[Dataset]]) -> Dict[str, int]:
    return {
        task_name: sum(len(dataset) for dataset in datasets)
        for task_name, datasets in task_groups.items()
        if datasets
    }
