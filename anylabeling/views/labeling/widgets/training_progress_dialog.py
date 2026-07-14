import csv
import os
import re

from PyQt5 import QtCore, QtGui, QtWidgets


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPOCH_HEADER_RE = re.compile(
    r"^\s*Epoch\s+GPU_mem\s+box_loss\s+obj_loss\s+cls_loss\s+Instances\s+Size"
)
EPOCH_ROW_RE = re.compile(r"^\s*\d+/\d+\s+\S+\s+[-+]?\d")


def clean_training_text(text):
    text = ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


def training_summary_lines(text):
    lines = []
    for line in clean_training_text(text).splitlines():
        if EPOCH_HEADER_RE.search(line) or EPOCH_ROW_RE.search(line):
            lines.append(line)
    return lines


def decode_process_output(data):
    raw = bytes(data)
    for encoding in ("utf-8", "gbk", "mbcs"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class TrainingCurveWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.setMinimumHeight(280)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

    def set_rows(self, rows):
        self.rows = rows
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.fillRect(self.rect(), QtGui.QColor("#f8fafc"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1))
        painter.drawRoundedRect(rect, 6, 6)

        if not self.rows:
            painter.setPen(QtGui.QColor("#64748b"))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "Waiting for results.csv")
            return

        panels = [
            ("Train Loss", ["train/box_loss", "train/obj_loss", "train/cls_loss"]),
            ("mAP50", ["metrics/mAP_0.5"]),
            ("mAP50-90", ["metrics/mAP_0.5:0.95"]),
        ]
        gap = 12
        panel_width = (rect.width() - gap * (len(panels) - 1)) / len(panels)
        for index, (title, keys) in enumerate(panels):
            left = int(rect.left() + index * (panel_width + gap))
            panel = QtCore.QRect(
                left,
                rect.top(),
                int(panel_width),
                rect.height(),
            )
            self.paint_panel(painter, panel, title, keys)

    def paint_panel(self, painter, rect, title, keys):
        painter.fillRect(rect.adjusted(1, 1, -1, -1), QtGui.QColor("#ffffff"))
        painter.setPen(QtGui.QColor("#0f172a"))
        title_rect = QtCore.QRect(rect.left() + 8, rect.top() + 6, rect.width() - 16, 22)
        painter.drawText(title_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, title)

        plot = rect.adjusted(58, 36, -12, -32)
        painter.setPen(QtGui.QPen(QtGui.QColor("#e2e8f0"), 1))
        for i in range(5):
            y = plot.top() + i * plot.height() / 4
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))
        painter.drawRect(plot)

        series = []
        for key in keys:
            values = [
                row[key]
                for row in self.rows
                if key in row and row[key] is not None
            ]
            if values:
                series.append((key, values))

        if not series:
            painter.setPen(QtGui.QColor("#94a3b8"))
            painter.drawText(plot, QtCore.Qt.AlignCenter, "No data")
            return

        all_values = [value for _, values in series for value in values]
        min_value = min(all_values)
        max_value = max(all_values)
        if max_value == min_value:
            max_value = min_value + 1.0

        painter.setPen(QtGui.QColor("#64748b"))
        font = painter.font()
        font.setPointSize(max(7, font.pointSize() - 1))
        painter.setFont(font)
        for ratio, value in (
            (1.0, max_value),
            (0.5, (min_value + max_value) / 2.0),
            (0.0, min_value),
        ):
            y = plot.bottom() - ratio * plot.height()
            label = f"{value:.3g}"
            label_rect = QtCore.QRect(
                rect.left() + 4,
                int(y) - 8,
                plot.left() - rect.left() - 8,
                16,
            )
            painter.drawText(
                label_rect,
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                label,
            )

        colors = ["#2563eb", "#16a34a", "#dc2626"]
        for index, (key, values) in enumerate(series):
            color = QtGui.QColor(colors[index % len(colors)])
            painter.setPen(QtGui.QPen(color, 2))
            points = []
            for point_index, value in enumerate(values):
                x_ratio = point_index / max(1, len(values) - 1)
                y_ratio = (value - min_value) / (max_value - min_value)
                x = plot.left() + x_ratio * plot.width()
                y = plot.bottom() - y_ratio * plot.height()
                points.append(QtCore.QPointF(x, y))
            if len(points) == 1:
                painter.drawEllipse(points[0], 2.5, 2.5)
            else:
                painter.drawPolyline(QtGui.QPolygonF(points))

            legend_y = rect.bottom() - 24 + index * 12
            painter.setPen(color)
            painter.drawText(rect.left() + 10, legend_y, self.short_metric_name(key))

        painter.setPen(QtGui.QColor("#64748b"))
        painter.drawText(
            rect.right() - 70,
            rect.bottom() - 10,
            f"{len(self.rows)} epochs",
        )

    def short_metric_name(self, key):
        return key.replace("metrics/", "").replace("train/", "").replace("val/", "")


class TrainingProgressDialog(QtWidgets.QDialog):
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, results_csv, parent=None):
        super().__init__(parent)
        self.results_csv = results_csv
        self.setWindowTitle(self.tr("YOLOv5 Training"))
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.resize(980, 680)
        self.setMinimumSize(760, 520)
        self.training_finished = False
        self.close_after_stop = False
        self.summary_header = ""
        self.epoch_rows = {}
        self.start_time = QtCore.QElapsedTimer()
        self.start_time.start()

        self.status_label = QtWidgets.QLabel(self.tr("Starting local training..."), self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "QLabel { font-weight: 600; color: #0f172a; padding: 6px 0; }"
        )

        self.curve_widget = TrainingCurveWidget(self)
        self.log_edit = QtWidgets.QPlainTextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(3000)
        self.log_edit.setStyleSheet(
            "QPlainTextEdit { background: #0f172a; color: #e2e8f0; "
            "font-family: Consolas, 'Courier New', monospace; font-size: 10pt; "
            "border-radius: 6px; padding: 8px; }"
        )

        self.cancel_button = QtWidgets.QPushButton(self.tr("Stop"), self)
        self.close_button = QtWidgets.QPushButton(self.tr("Close"), self)
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.confirm_stop)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.close_button)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        curve_container = QtWidgets.QWidget(self)
        curve_layout = QtWidgets.QVBoxLayout(curve_container)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        curve_layout.addWidget(self.curve_widget)
        splitter.addWidget(curve_container)
        splitter.addWidget(self.log_edit)
        splitter.setSizes([360, 240])

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(splitter, 1)
        layout.addLayout(button_layout)

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(1500)
        self.poll_timer.timeout.connect(self.refresh_curves)
        self.poll_timer.start()

    def set_status(self, text):
        self.status_label.setText(text)

    def append_output(self, text):
        lines = training_summary_lines(text)
        if not lines:
            return
        last_line = ""
        for line in lines:
            if EPOCH_HEADER_RE.search(line):
                self.summary_header = line
                last_line = line
                continue
            match = re.match(r"^\s*(\d+/\d+)\s+", line)
            if match:
                self.epoch_rows[match.group(1)] = line
                last_line = line

        display_lines = []
        if self.summary_header:
            display_lines.append(self.summary_header)
        display_lines.extend(self.epoch_rows.values())
        self.log_edit.setPlainText("\n".join(display_lines))
        self.log_edit.moveCursor(QtGui.QTextCursor.End)
        self.status_label.setText(last_line[-260:])

    def refresh_curves(self):
        rows = self.read_results_csv()
        if rows:
            self.curve_widget.set_rows(rows)
            self.update_summary_from_csv(rows)
        elif not self.training_finished:
            elapsed = max(1, self.start_time.elapsed() // 1000)
            self.status_label.setText(
                self.tr(
                    "Initializing local training... %ss elapsed. "
                    "First run may scan or repair images before epoch output appears."
                )
                % elapsed
            )

    def read_results_csv(self):
        if not os.path.isfile(self.results_csv):
            return []
        rows = []
        try:
            with open(self.results_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        key = key.strip()
                        try:
                            parsed[key] = float(str(value).strip())
                        except (TypeError, ValueError):
                            parsed[key] = None
                    if parsed:
                        rows.append(parsed)
        except OSError:
            return []
        return rows

    def update_summary_from_csv(self, rows):
        if not rows:
            return
        if not self.summary_header:
            self.summary_header = (
                "      Epoch    GPU_mem   box_loss   obj_loss   "
                "cls_loss  Instances       Size"
            )
        for row in rows:
            epoch = row.get("epoch")
            if epoch is None:
                continue
            epoch_key = str(int(epoch))
            self.epoch_rows[epoch_key] = (
                f"{epoch_key:>11} {'-':>10} "
                f"{self.format_metric(row.get('train/box_loss')):>10} "
                f"{self.format_metric(row.get('train/obj_loss')):>10} "
                f"{self.format_metric(row.get('train/cls_loss')):>10} "
                f"{'-':>10} {'-':>10}"
            )

        display_lines = [self.summary_header, *self.epoch_rows.values()]
        self.log_edit.setPlainText("\n".join(display_lines))
        self.log_edit.moveCursor(QtGui.QTextCursor.End)
        latest = rows[-1]
        latest_epoch = latest.get("epoch")
        if latest_epoch is not None:
            self.status_label.setText(
                self.tr("Epoch %s completed. mAP50=%s, mAP50-90=%s")
                % (
                    int(latest_epoch),
                    self.format_metric(latest.get("metrics/mAP_0.5")),
                    self.format_metric(latest.get("metrics/mAP_0.5:0.95")),
                )
            )

    def format_metric(self, value):
        if value is None:
            return "-"
        return f"{value:.5g}"

    def finish(self, success, message):
        self.training_finished = True
        self.poll_timer.stop()
        self.refresh_curves()
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.status_label.setText(message)
        if success:
            self.status_label.setStyleSheet(
                "QLabel { font-weight: 600; color: #166534; padding: 6px 0; }"
            )
        else:
            self.status_label.setStyleSheet(
                "QLabel { font-weight: 600; color: #b91c1c; padding: 6px 0; }"
            )
        if self.close_after_stop:
            self.accept()

    def confirm_stop(self):
        if self.training_finished:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Stop Training"),
            self.tr("Training is still running. Stop it now?"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.status_label.setText(self.tr("Stopping training..."))
        self.cancel_button.setEnabled(False)
        self.stop_requested.emit()

    def closeEvent(self, event):
        if self.training_finished:
            event.accept()
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Stop Training"),
            self.tr("Training is still running. Closing this window will stop it. Continue?"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            event.ignore()
            return
        self.close_after_stop = True
        self.cancel_button.setEnabled(False)
        self.stop_requested.emit()
        event.ignore()
