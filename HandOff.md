# 工程新增特点

1. 数据目录本地化：默认数据从用户目录调整为工程内 `xanylabeling_data`，便于项目整体迁移和管理；仍支持 `XANYLABELING_DATA_DIR` 环境变量覆盖。
2. 图像批量切分：左侧文件面板新增切图入口，可按行列数切分当前打开文件夹内所有图像，原图移动到上一级 `<原文件夹>_raw`，当前目录替换为切分后小图。
3. 切图异步化：切图流程放入 `QThread` 后台执行，并显示进度弹窗，避免大批量图片处理时界面卡死。
4. 标注导出 YOLO：新增 X-AnyLabeling JSON 转 YOLO 数据集脚本和导出默认路径逻辑，默认输入为当前打开目录，输出为 `<目录名>-yolo_dataset`。
5. ONNX 类别自动读取：YOLO 自动标注和 JSON 转 YOLO 导出支持从 ONNX metadata 的 `names` 中读取类别，减少手写 `classes` 配置。
6. 自定义模型配置：自动 AI 标注面板新增“Configure Model”按钮，可配置 YOLO 类型、模型名、ONNX 路径、IOU/置信度阈值，并生成自定义模型 YAML 后直接加载。
7. 训练规划入口：提示词文档已补充 ClearML/服务器训练需求，明确后续重点是用服务器 B 的 4090D/5090 做训练，笔记本主要用于标注、验证和轻量推理。

主要代码位置：

- `anylabeling/paths.py`
- `anylabeling/views/labeling/label_widget.py`
- `anylabeling/views/labeling/utils/export.py`
- `anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.py`
- `anylabeling/services/auto_labeling/__base__/yolo.py`
- `scripts/xanylabeling_json_to_yolo.py`
