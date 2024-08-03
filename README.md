# ResNet_NET
ResNet_NET 项目包含两个核心部分:预训练ResNet模型和自定义图像分类模型。


ResNet_NET: .NET 环境下的 ResNet 图像分类解决方案
项目介绍
ResNet_NET 是一个基于 .NET 平台的综合图像分类解决方案，专注于使用 ResNet 系列模型进行图像识别。本解决方案包含两个主要项目：

ResNet ONNX 实时分类器
ML.NET ResNet 自定义训练器
这两个项目共同提供了一个完整的 ResNet 模型应用和训练流程，适用于不同的使用场景。

1. ResNet ONNX 实时分类器
该项目使用预训练的 ResNet152 v2-7 ONNX 模型，提供快速、高效的实时图像分类功能。

特性：

使用高性能的 ResNet152 v2-7 预训练模型
支持实时图像分类
集成中文标签支持
简单的命令行界面，易于使用
2. ML.NET ResNet 自定义训练器
这个项目利用 ML.NET 框架，允许用户使用自己的数据集训练和测试 ResNet 模型。

特性：

基于 ML.NET 框架
支持自定义数据集的模型训练
使用 ResNet 架构进行图像分类
包含模型评估和测试功能
部署和使用方法
环境要求
.NET 6.0 或更高版本
Visual Studio 2019 或更高版本（推荐），或其他支持 .NET 的 IDE
项目设置
克隆仓库：

git clone https://github.com/zou280/ResNet_NET.git
使用 Visual Studio 打开解决方案文件。
ResNet ONNX 实时分类器使用方法
下载必要文件：
ResNet152 v2 ONNX 模型:https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet152-v2-7.onnx
将下载的文件放在项目根目录。
运行程序,根据提示输入图片路径进行分类。
输入 "exit" 退出程序。
ML.NET ResNet 自定义训练器使用方法
准备数据集：
在项目根目录创建 训练数据集 文件夹，按类别创建子文件夹并放置对应的图像文件和 images.tsv 文件（包含图像路径和标签信息）
在项目根目录创建 测试数据集 文件夹，按类别创建子文件夹并放置对应的图像文件
运行程序,
在程序提示时：
选择 "1" 进行模型训练
选择 "2" 进行模型测试
注意事项
确保安装了所有必要的 NuGet 包：ML.NET、SixLabors.ImageSharp 和 Microsoft.ML.OnnxRuntime。
ResNet ONNX 项目：确保模型文件和标签文件放置在正确的位置。
ML.NET 项目：确保训练数据集格式正确，测试数据集按类别正确组织。
ONNX 模型文件较大，请确保有足够的存储空间和下载带宽。
贡献
我们欢迎各种形式的贡献，包括但不限于：

提交 bug 报告
新功能建议
代码改进和 Pull Requests
许可
本项目采用 MIT 许可证。

联系方式
如有任何问题或建议，请通过 GitHub Issues 与我们联系。
