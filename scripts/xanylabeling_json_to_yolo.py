import argparse
import json
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classes_from_yaml(yaml_path: Path) -> list[str]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"No valid 'classes' found in {yaml_path}")
    return [str(x) for x in classes]


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
    else:
        parts.insert(0, "labels")
    return output_root.joinpath(*parts).with_suffix(".txt")


def build_output_image_path(
    json_file: Path, input_root: Path, output_root: Path, image_suffix: str
) -> Path:
    relative = json_file.relative_to(input_root)
    parts = list(relative.parts)
    if parts and parts[0] == "labels":
        parts[0] = "images"
    elif parts and parts[0] != "images":
        parts.insert(0, "images")
    elif not parts:
        parts = ["images", json_file.name]
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

    print(f"json files: {len(json_files)}")
    print(f"classes: {len(classes)} -> {classes}")
    print(f"converted shapes: {converted_total}")
    print(f"skipped shapes: {skipped_total}")
    if not val_dir.exists():
        print("val split not found; dataset.yaml uses images/train as val")
    print(f"output root: {output_root}")


if __name__ == "__main__":
    main()
