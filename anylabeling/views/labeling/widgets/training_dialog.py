import os
import os.path as osp
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


DEFAULT_YOLOV5_DIR = (
    r"D:\Work_mumu\LabelAndTrain\x-train\yolov5-7.0"
)
DEFAULT_YOLOV5_PYTHON = r"D:\Anaconda3\envs\yolov5-7\python.exe"
DEFAULT_WORKERS = 0 if os.name == "nt" else 8


class TrainingDialog(QDialog):
    """Collect parameters for local YOLOv5 training."""

    def __init__(self, settings, last_open_dir=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle(self.tr("Train YOLOv5"))
        self.setMinimumWidth(720)

        self.python_input = self.path_input("training/python")
        self.python_input.setText(
            self.python_input.text() or DEFAULT_YOLOV5_PYTHON
        )
        self.yolov5_dir_input = self.path_input("training/yolov5_dir")
        self.yolov5_dir_input.setText(
            self.yolov5_dir_input.text() or DEFAULT_YOLOV5_DIR
        )

        default_data = self.default_dataset_yaml(last_open_dir)
        self.data_input = self.path_input("training/data_yaml")
        self.data_input.setText(self.data_input.text() or default_data)

        self.result_dir_input = self.path_input("training/result_dir")
        default_result_dir = self.default_result_dir(self.data_input.text())
        if self.should_use_dataset_default(
            self.result_dir_input.text(), default_result_dir
        ):
            self.result_dir_input.setText(default_result_dir)

        self.hyp_input = self.path_input("training/hyp_yaml")
        default_hyp_yaml = self.default_hyp_yaml(self.data_input.text())
        if self.should_use_dataset_default(
            self.hyp_input.text(), default_hyp_yaml
        ):
            self.hyp_input.setText(default_hyp_yaml)
        self.weights_input = QLineEdit(
            settings.value("training/weights", "yolov5s.pt", type=str), self
        )
        self.device_input = QLineEdit(
            settings.value("training/device", "0", type=str), self
        )
        self.extra_args_input = QLineEdit(
            settings.value("training/extra_args", "", type=str), self
        )

        self.epochs_spin = QSpinBox(self)
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(
            settings.value("training/epochs", 100, type=int)
        )
        self.batch_spin = QSpinBox(self)
        self.batch_spin.setRange(-1, 1024)
        self.batch_spin.setValue(
            settings.value("training/batch_size", 16, type=int)
        )
        self.imgsz_spin = QSpinBox(self)
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(
            settings.value("training/imgsz", 640, type=int)
        )
        self.workers_spin = QSpinBox(self)
        self.workers_spin.setRange(0, 64)
        workers = settings.value(
            "training/workers", DEFAULT_WORKERS, type=int
        )
        if os.name == "nt" and workers == 8:
            workers = 0
        self.workers_spin.setValue(workers)

        self.data_input.textChanged.connect(self.update_result_dir)
        self.data_input.textChanged.connect(self.update_hyp_yaml)

        form = QFormLayout()
        form.addRow(
            self.tr("Python"),
            self.with_browse(self.python_input, self.browse_python),
        )
        form.addRow(
            self.tr("YOLOv5 dir"),
            self.with_browse(self.yolov5_dir_input, self.browse_yolov5_dir),
        )
        form.addRow(
            self.tr("dataset yaml"),
            self.with_browse(self.data_input, self.browse_data_yaml),
        )
        form.addRow(
            self.tr("result dir"),
            self.with_browse(self.result_dir_input, self.browse_result_dir),
        )
        form.addRow(
            self.tr("hyp yaml"),
            self.with_browse(self.hyp_input, self.browse_hyp_yaml),
        )
        form.addRow(self.tr("weights"), self.weights_input)
        form.addRow(self.tr("device"), self.device_input)
        form.addRow(self.tr("epochs"), self.epochs_spin)
        form.addRow(self.tr("batch size"), self.batch_spin)
        form.addRow(self.tr("image size"), self.imgsz_spin)
        form.addRow(self.tr("workers"), self.workers_spin)
        form.addRow(self.tr("extra train args"), self.extra_args_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def path_input(self, key):
        return QLineEdit(self.settings.value(key, "", type=str), self)

    def with_browse(self, line_edit, callback):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton(self.tr("Browse"), self)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return layout

    def browse_python(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select Python"), "", self.tr("Python (*.exe)")
        )
        if path:
            self.python_input.setText(osp.normpath(path))

    def browse_yolov5_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, self.tr("Select YOLOv5 Directory"), self.yolov5_dir_input.text()
        )
        if path:
            self.yolov5_dir_input.setText(osp.normpath(path))

    def browse_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Dataset YAML"),
            osp.dirname(self.data_input.text()) or "",
            self.tr("YAML (*.yaml *.yml)"),
        )
        if path:
            self.data_input.setText(osp.normpath(path))

    def browse_result_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Result Directory"),
            self.result_dir_input.text() or "",
        )
        if path:
            self.result_dir_input.setText(osp.normpath(path))

    def browse_hyp_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Hyperparameter YAML"),
            osp.dirname(self.hyp_input.text()) or "",
            self.tr("YAML (*.yaml *.yml)"),
        )
        if path:
            self.hyp_input.setText(osp.normpath(path))

    def update_result_dir(self):
        default_result_dir = self.default_result_dir(self.data_input.text())
        if self.should_use_dataset_default(
            self.result_dir_input.text(), default_result_dir
        ):
            self.result_dir_input.setText(default_result_dir)

    def update_hyp_yaml(self):
        default_hyp_yaml = self.default_hyp_yaml(self.data_input.text())
        if self.should_use_dataset_default(
            self.hyp_input.text(), default_hyp_yaml
        ):
            self.hyp_input.setText(default_hyp_yaml)

    def default_dataset_yaml(self, last_open_dir):
        if not last_open_dir:
            return ""
        folder = Path(last_open_dir)
        candidates = [
            folder / "dataset.yaml",
            folder / "data.yaml",
            folder.parent / f"{folder.name}-yolo_dataset" / "dataset.yaml",
            folder.parent / f"{folder.name}_yolo_dataset" / "dataset.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return ""

    def default_result_dir(self, data_yaml):
        if not data_yaml:
            return ""
        return str(Path(data_yaml).expanduser().resolve().parent / "train_result")

    def default_hyp_yaml(self, data_yaml):
        if not data_yaml:
            return ""
        return str(Path(data_yaml).expanduser().resolve().parent / "hyp.yaml")

    def should_use_dataset_default(self, current_path, default_path):
        if not default_path:
            return False
        if not current_path:
            return True
        try:
            current = Path(current_path).expanduser().resolve()
            default = Path(default_path).expanduser().resolve()
        except OSError:
            return True
        return current.parent != default.parent

    def accept(self):
        error = self.validate_inputs()
        if error:
            QMessageBox.warning(self, self.tr("Train YOLOv5"), error)
            return
        self.save_settings()
        super().accept()

    def validate_inputs(self):
        python_path = self.python_input.text().strip()
        yolov5_dir = self.yolov5_dir_input.text().strip()
        data_yaml = self.data_input.text().strip()
        result_dir = self.result_dir_input.text().strip()
        hyp_yaml = self.hyp_input.text().strip()

        if not osp.isfile(python_path):
            return self.tr("Python executable does not exist.")
        if not osp.isfile(osp.join(yolov5_dir, "train.py")):
            return self.tr("YOLOv5 directory must contain train.py.")
        if not osp.isfile(data_yaml):
            return self.tr("Dataset yaml cannot be empty.")
        if not result_dir:
            return self.tr("Result directory cannot be empty.")
        if not osp.isfile(hyp_yaml):
            return self.tr("Hyperparameter yaml does not exist.")
        return ""

    def save_settings(self):
        values = self.training_config()
        self.settings.setValue("training/python", values["python"])
        self.settings.setValue("training/yolov5_dir", values["yolov5_dir"])
        self.settings.setValue("training/data_yaml", values["data"])
        self.settings.setValue("training/result_dir", values["result_dir"])
        self.settings.setValue("training/hyp_yaml", values["hyp"])
        self.settings.setValue("training/weights", values["weights"])
        self.settings.setValue("training/device", values["device"])
        self.settings.setValue("training/epochs", values["epochs"])
        self.settings.setValue("training/batch_size", values["batch_size"])
        self.settings.setValue("training/imgsz", values["imgsz"])
        self.settings.setValue("training/workers", values["workers"])
        self.settings.setValue("training/extra_args", values["extra_args"])

    def training_config(self):
        return {
            "python": self.python_input.text().strip(),
            "yolov5_dir": self.yolov5_dir_input.text().strip(),
            "data": self.data_input.text().strip(),
            "result_dir": self.result_dir_input.text().strip(),
            "hyp": self.hyp_input.text().strip(),
            "weights": self.weights_input.text().strip(),
            "device": self.device_input.text().strip(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "workers": self.workers_spin.value(),
            "extra_args": self.extra_args_input.text().strip(),
        }
