using System;
using System.Data;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using Microsoft.ML.Transforms;
using Tensorflow.Contexts;
using static Program;
using Microsoft.ML.Transforms.Image;
using SkiaSharp;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Text; // 添加这一行

class Program
{


    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }

    }

    public class ImageDataWithFeaturesByte : ImageData
    {
        [ColumnName("LabelKey")]
        public uint LabelKey { get; set; }

        [ColumnName("FeaturesByte")]
        public byte[] FeaturesByte { get; set; }
    }


    // 定义用于预测的类
    public class ImagePrediction : ImageDataWithFeaturesByte
    {
        public float[] Score { get; set; } //分数
        public uint PredictedLabel { get; set; }//预测标签
    }

    // 创建 MLContext
    static  MLContext mlContext = new MLContext();

    static string directory = "";

    public static void 训练()
    {
        var mlContext = new MLContext();

        // 加载数据
        IDataView data = mlContext.Data.LoadFromTextFile<ImageData>(
            path: directory+"训练数据集\\images.tsv",
            hasHeader: true,
            separatorChar: ' ');

        // 将图像路径转换为字节数组
        var featureBytesData = mlContext.Transforms.CustomMapping<ImageData, ImageDataWithFeaturesByte>((input, output) =>
        {
            output.FeaturesByte = File.ReadAllBytes(input.ImagePath);
            output.Label = input.Label;
        }, contractName: null)
        .Fit(data)
        .Transform(data);

        // 将标签字符串转换为无符号整数键值
        var labelEncodingPipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelKey", nameof(ImageData.Label))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("LabelKeyEncoded", "LabelKey"));

        var labeledData = labelEncodingPipeline.Fit(featureBytesData).Transform(featureBytesData);

        // 检查编码后的标签
        Console.WriteLine("检查编码后的标签:");
        var labelColumn = labeledData.GetColumn<uint>("LabelKeyEncoded").ToArray();
        var uniqueLabels = labelColumn.Distinct().ToArray();
        Console.WriteLine($"唯一编码标签数量: {uniqueLabels.Length}");
        Console.WriteLine($"唯一编码标签: {string.Join(", ", uniqueLabels)}");

        if (uniqueLabels.Length < 2)
        {
            throw new Exception("编码后的唯一标签数量不足。请检查您的数据集和编码过程。");
        }

        // 打乱数据行
        IDataView shuffledDataView = mlContext.Data.ShuffleRows(labeledData);

        // 分割数据集
        var trainTestData = mlContext.Data.TrainTestSplit(shuffledDataView, testFraction: 0.2);

        // 检查训练集中的标签分布
        Console.WriteLine("\n检查训练集中的标签分布:");
        var trainSet = trainTestData.TrainSet;
        var labelDistribution = mlContext.Data.CreateEnumerable<ImageDataWithFeaturesByte>(trainSet, reuseRowObject: false)
            .GroupBy(x => x.Label)
            .Select(g => new { Label = g.Key, Count = g.Count() })
            .OrderBy(x => x.Label)
            .ToList();

        foreach (var item in labelDistribution)
        {
            Console.WriteLine($"标签 {item.Label}: {item.Count} 个样本");
        }

        var pipeline = mlContext.Transforms.LoadImages(
                outputColumnName: "Image",
                imageFolder: directory + "训练数据集",
                inputColumnName: nameof(ImageData.ImagePath))
            .Append(mlContext.Transforms.ResizeImages(
                outputColumnName: "Image",
                imageWidth: 224,
                imageHeight: 224,
                resizing: ImageResizingEstimator.ResizingKind.Fill))
            .Append(mlContext.Transforms.ExtractPixels(
                outputColumnName: "Features",
                inputColumnName: "Image"))
            .Append(mlContext.Transforms.NormalizeMinMax(
                outputColumnName: "Features",
                inputColumnName: "Features"))
            .AppendCacheCheckpoint(mlContext);

        // 训练模型
        Console.WriteLine("正在训练模型...");
        ITransformer model = pipeline.Fit(trainTestData.TrainSet);

        var imageClassificationTrainer = mlContext.MulticlassClassification.Trainers.ImageClassification(new ImageClassificationTrainer.Options
        {
            FeatureColumnName = "FeaturesByte",
            LabelColumnName = "LabelKeyEncoded",
            ValidationSet = trainTestData.TestSet,
            Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
            MetricsCallback = (metrics) => Console.WriteLine(metrics),
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true,
        });

        // 创建包含图像分类训练器的新管道
        var trainingPipeline = pipeline.Append(imageClassificationTrainer);

        // 使用新管道重新拟合模型
        model = trainingPipeline.Fit(trainTestData.TrainSet);

        // 评估模型
        Console.WriteLine("正在评估模型...");
        IDataView predictions = model.Transform(trainTestData.TestSet);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKeyEncoded", scoreColumnName: "Score");
        Console.WriteLine($"宏观准确率: {metrics.MacroAccuracy:F2}");
        Console.WriteLine($"微观准确率: {metrics.MicroAccuracy:F2}");
        Console.WriteLine($"对数损失: {metrics.LogLoss:F2}");

        // 预览一些预测结果
        Console.WriteLine("\n预览预测结果:");
        var preview = mlContext.Data.CreateEnumerable<ImageDataWithFeaturesByte>(predictions, reuseRowObject: false).Take(5);
        foreach (var row in preview)
        {
            var scoreColumn = predictions.Schema["Score"];
            var score = predictions.GetColumn<float[]>(scoreColumn).First();
            Console.WriteLine($"标签: {row.Label}, 分数: [{string.Join(", ", score)}]");
        }

        // 保存模型
        mlContext.Model.Save(model, shuffledDataView.Schema, "model.zip");
        Console.WriteLine("模型已训练并保存。");
    }



    public static void 测试()
    {

        string folderPath = directory + "测试数据集"; // 替换为你的文件夹路径

        // 定义图片文件的扩展名
        string[] imageExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp"};

        // 获取文件夹下所有图片文件,   // 待预测的图像路径
        var imageFiles = Directory.GetFiles(folderPath, "*.*", SearchOption.AllDirectories)
                                   .Where(file => imageExtensions.Contains(Path.GetExtension(file).ToLower())).ToArray().Take(10)
                                   .ToArray();



        var mlContext = new MLContext();

        // 加载模型
        ITransformer trainedModel;
        DataViewSchema modelSchema;
        using (var fileStream = new FileStream("model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            trainedModel = mlContext.Model.Load(fileStream, out modelSchema);
        }

        // 创建预测引擎
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageDataWithFeaturesByte, ImagePrediction>(trainedModel);

 
        int de_count = 0;//预测正确数量

        foreach (var imagePath in imageFiles)
        {

            FileInfo fileInfo = new FileInfo(imagePath);

            // 获取父文件夹信息
            DirectoryInfo parentDirectory = fileInfo.Directory;

            // 获取父文件夹名称
            string parentDirectoryName = parentDirectory.Name;
       
               // 读取图像并进行预测
               var imageData = new ImageDataWithFeaturesByte
            {
                ImagePath = imagePath,
                Label= parentDirectoryName,
                FeaturesByte = File.ReadAllBytes(imagePath)
            };

            var prediction = predictionEngine.Predict(imageData);
            prediction.PredictedLabel = prediction.PredictedLabel - 1;

            if (prediction.PredictedLabel.ToString()== imageData.Label) {

                de_count++;
            }
            // 输出预测结果
            Console.WriteLine($"图像: {imagePath}");
            Console.WriteLine($"预测标签: {prediction.PredictedLabel}");
            Console.WriteLine($"分数: [{string.Join(", ", prediction.Score)}]");
        }
        decimal 正确率 = (decimal)de_count / (decimal)imageFiles.Length;
        Console.WriteLine($"正确数量："+ de_count);
        Console.WriteLine($"正确率：" + 正确率);
    }






    static void Main(string[] args)
    {
        directory = AppDomain.CurrentDomain.BaseDirectory;

        Console.Write("1=>训练，2=>测试 \n");
        string Image_path = Console.ReadLine().Replace("?", " ").Trim();

        switch (Image_path.ToLower())
        {
            case "1":
                训练();
                break;

            case "2":
                测试();
                break;

        }
    }
}