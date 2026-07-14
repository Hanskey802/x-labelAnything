# 工程新增特点

1. 数据目录本地化：默认数据从用户目录调整为工程内 `xanylabeling_data`，便于项目整体迁移和管理；仍支持 `XANYLABELING_DATA_DIR` 环境变量覆盖。
2. 图像批量切分：左侧文件面板新增切图入口，可按行列数切分当前打开文件夹内所有图像，原图移动到上一级 `<原文件夹>_raw`，当前目录替换为切分后小图。
3. 切图异步化：切图流程放入 `QThread` 后台执行，并显示进度弹窗，避免大批量图片处理时界面卡死。
4. 标注导出 YOLO：新增 X-AnyLabeling JSON 转 YOLO 数据集脚本和导出默认路径逻辑，默认输入为当前打开目录，输出为 `<目录名>-yolo_dataset`，无显式划分时默认生成 `images/train` 与 `labels/train`，同时生成默认 `hyp.yaml`。
5. ONNX 类别自动读取：YOLO 自动标注和 JSON 转 YOLO 导出支持从 ONNX metadata 的 `names` 中读取类别，减少手写 `classes` 配置。
6. 自定义模型配置：自动 AI 标注面板新增“Configure Model”按钮，可配置 YOLO 类型、模型名、ONNX 路径、IOU/置信度阈值，并生成自定义模型 YAML 后直接加载。
7. 训练规划入口：提示词文档已补充服务器/本地训练需求，明确后续重点是用服务器 B 的 4090D/5090 做主训练，笔记本主要用于标注、验证和轻量训练。
8. YOLOv5 本地训练入口：顶部菜单栏新增“训练”按钮，通过参数弹窗直接启动本机 YOLOv5 `train.py`；默认读取数据集目录的 `hyp.yaml`，训练结果统一写入数据集目录的 `train_result/exp`。
9. 训练过程可视化：训练窗口设置 UTF-8 输出环境，清理 ANSI 控制字符；日志只保留 epoch 汇总行，并实时读取 `results.csv` 绘制 train loss、mAP50、mAP50-90 曲线，图表显示纵坐标数值。窗口支持最小化，训练中关闭会提示是否停止训练。
10. YOLOv5 CUDA 环境：已创建 `D:\Anaconda3\envs\yolov5-7`，安装 `torch 1.13.1+cu117`、`torchvision 0.14.1+cu117`，本机 RTX 3060 可被 PyTorch 识别；`setuptools` 固定为 `65.5.1`，避免 YOLOv5 7.0 导入 `pkg_resources` 时出现弃用告警。

主要代码位置：

- `anylabeling/paths.py`
- `anylabeling/views/labeling/label_widget.py`
- `anylabeling/views/labeling/utils/export.py`
- `anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.py`
- `anylabeling/views/labeling/widgets/training_dialog.py`
- `anylabeling/views/labeling/widgets/training_progress_dialog.py`
- `anylabeling/services/auto_labeling/__base__/yolo.py`
- `scripts/xanylabeling_json_to_yolo.py`
