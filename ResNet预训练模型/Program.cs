using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

class ResNet50模型Program
{
    static string directory = "";

    static void Main(string[] args)
    {
        directory = AppDomain.CurrentDomain.BaseDirectory;


        // 加载预训练的 ONNX 模型
        string modelPath = directory+"resnet152-v2-7.onnx";
        using var session = new InferenceSession(modelPath);

        // 加载类别标签（假设你有一个包含类别标签的文本文件）
        string[] labels = File.ReadAllLines(directory + "标签文件(中文).txt");

        bool continueRunning = true;


        while (continueRunning)
        {
            Console.Write("请输入图片地址，（输入'exit'退出）: ");
            string Image_path = Console.ReadLine().Replace("?"," ").Trim();

            switch (Image_path.ToLower())
            {
                case "exit":
                    continueRunning = false;
                    break;
                default:
                 //   Console.WriteLine($"你输入的是: {Image_path}");
                     var image = Image.Load<Rgb24>(Image_path);

                    // 预处理图像
                    image.Mutate(x => x.Resize(224, 224));
                    var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
                    for (int y = 0; y < image.Height; y++)
                    {
                        for (int x = 0; x < image.Width; x++)
                        {
                            input[0, 0, y, x] = (image[x, y].R / 255f - 0.485f) / 0.229f;
                            input[0, 1, y, x] = (image[x, y].G / 255f - 0.456f) / 0.224f;
                            input[0, 2, y, x] = (image[x, y].B / 255f - 0.406f) / 0.225f;
                        }
                    }

                    // 运行推理
                    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data", input) };
                    var results = session.Run(inputs);

                    // 处理结果
                    var output = results.First().AsEnumerable<float>().ToArray();
                    var maxIndex = Array.IndexOf(output, output.Max());


                    Console.WriteLine($"识别结果: {labels[maxIndex]}");
                    break;
            }
        }

       // Console.ReadKey();

    }
}