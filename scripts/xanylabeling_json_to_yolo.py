import argparse
import ast
import json
import shutil
from pathlib import Path

import onnx
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_SPLITS = {"train", "val", "test"}
DEFAULT_HYPERPARAMETERS = {
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 0.05,
    "cls": 0.5,
    "cls_pw": 1.0,
    "obj": 1.0,
    "obj_pw": 1.0,
    "iou_t": 0.20,
    "anchor_t": 4.0,
    "fl_gamma": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


def load_classes_from_yaml(yaml_path: Path) -> list[str]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    classes = data.get("classes")
    if isinstance(classes, list) and classes:
        return [str(x) for x in classes]

    model_path = data.get("model_path")
    if not model_path:
        raise ValueError(
            f"No valid 'classes' or 'model_path' found in {yaml_path}"
        )

    model_path = Path(str(model_path)).expanduser()
    if not model_path.is_absolute():
        model_path = (yaml_path.parent / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    return load_classes_from_onnx(model_path)


def load_classes_from_onnx(model_path: Path) -> list[str]:
    model = onnx.load(str(model_path))
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    names = metadata.get("names")
    if not names:
        raise ValueError(f"No ONNX metadata 'names' found in {model_path}")

    try:
        parsed = ast.literal_eval(names)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse ONNX metadata 'names' in {model_path}"
        ) from exc

    if isinstance(parsed, dict):
        try:
            items = sorted(parsed.items(), key=lambda item: int(item[0]))
        except Exception:
            items = parsed.items()
        return [str(value) for _, value in items]

    if isinstance(parsed, (list, tuple)):
        return [str(value) for value in parsed]

    raise ValueError(f"Unsupported ONNX metadata 'names' format in {model_path}")


def discover_classes(json_files: list[Path]) -> list[str]:
    labels: set[str] = set()
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label:
                labels.add(str(label))
    return sorted(labels)


def normalize_rect(points, image_width: float, image_height: float):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height


def build_output_label_path(
    json_file: Path, input_root: Path, output_root: Path
) -> Path:
    relative = json_file.relative_to(input_root)
    parts = list(relative.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
    elif parts and parts[0] == "labels":
        pass
    elif parts and parts[0] in YOLO_SPLITS:
        parts.insert(0, "labels")
    else:
        parts = ["labels", "train", *parts]
    return output_root.joinpath(*parts).with_suffix(".txt")


def build_output_image_path(
    json_file: Path, input_root: Path, output_root: Path, image_suffix: str
) -> Path:
    relative = json_file.relative_to(input_root)
    parts = list(relative.parts)
    if parts and parts[0] == "labels":
        parts[0] = "images"
    elif parts and parts[0] == "images":
        pass
    elif parts and parts[0] in YOLO_SPLITS:
        parts.insert(0, "images")
    elif parts and parts[0] != "images":
        parts = ["images", "train", *parts]
    elif not parts:
        parts = ["images", "train", json_file.name]
    return output_root.joinpath(*parts).with_suffix(image_suffix)


def find_image_for_json(json_file: Path, image_path_value: str | None) -> Path | None:
    if image_path_value:
        candidate = json_file.parent / image_path_value
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTS:
        candidate = json_file.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def convert_one(
    json_file: Path,
    input_root: Path,
    output_root: Path,
    class_to_id: dict[str, int],
    write_empty: bool,
) -> tuple[int, int]:
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")
    if not image_width or not image_height:
        raise ValueError(f"Missing image size in {json_file}")

    rows: list[str] = []
    converted = 0
    skipped = 0

    for shape in data.get("shapes", []):
        label = shape.get("label")
        shape_type = shape.get("shape_type")
        points = shape.get("points", [])

        if label not in class_to_id:
            skipped += 1
            continue

        if shape_type != "rectangle":
            skipped += 1
            continue

        if len(points) < 2:
            skipped += 1
            continue

        x_center, y_center, width, height = normalize_rect(
            points, float(image_width), float(image_height)
        )
        class_id = class_to_id[label]
        rows.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
        converted += 1

    label_path = build_output_label_path(json_file, input_root, output_root)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if rows or write_empty:
        label_path.write_text("\n".join(rows), encoding="utf-8")

    image_file = find_image_for_json(json_file, data.get("imagePath"))
    if image_file:
        image_out = build_output_image_path(
            json_file, input_root, output_root, image_file.suffix
        )
        image_out.parent.mkdir(parents=True, exist_ok=True)
        if not image_out.exists():
            shutil.copy2(image_file, image_out)

    return converted, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert X-AnyLabeling JSON annotations to YOLO txt labels."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing json files, e.g. E:/dataset/yolo_xinda",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output dataset root, e.g. E:/dataset/yolo_xinda_yolo",
    )
    parser.add_argument(
        "--classes-yaml",
        help="Optional model yaml file whose 'classes' list defines YOLO class order.",
    )
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="Write empty .txt files for images whose json has no valid shapes.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    json_files = sorted(input_root.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No json files found under {input_root}")

    if args.classes_yaml:
        classes = load_classes_from_yaml(Path(args.classes_yaml).resolve())
    else:
        classes = discover_classes(json_files)

    if not classes:
        raise ValueError("No classes found.")

    class_to_id = {name: idx for idx, name in enumerate(classes)}

    converted_total = 0
    skipped_total = 0
    for json_file in json_files:
        converted, skipped = convert_one(
            json_file,
            input_root=input_root,
            output_root=output_root,
            class_to_id=class_to_id,
            write_empty=args.write_empty,
        )
        converted_total += converted
        skipped_total += skipped

    names_path = output_root / "classes.txt"
    names_path.write_text("\n".join(classes), encoding="utf-8")

    val_dir = output_root / "images" / "val"
    test_dir = output_root / "images" / "test"

    dataset_config = {
        "path": str(output_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val" if val_dir.exists() else "images/train",
        "names": {idx: name for idx, name in enumerate(classes)},
        "nc": len(classes),
    }
    if test_dir.exists():
        dataset_config["test"] = "images/test"

    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            dataset_config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    hyp_yaml = output_root / "hyp.yaml"
    hyp_yaml.write_text(
        yaml.safe_dump(
            DEFAULT_HYPERPARAMETERS,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(f"json files: {len(json_files)}")
    print(f"classes: {len(classes)} -> {classes}")
    print(f"converted shapes: {converted_total}")
    print(f"skipped shapes: {skipped_total}")
    if not val_dir.exists():
        print("val split not found; dataset.yaml uses images/train as val")
    print(f"output root: {output_root}")
    print(f"hyp yaml: {hyp_yaml}")


if __name__ == "__main__":
    main()
