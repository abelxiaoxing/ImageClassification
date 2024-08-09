using System;
using System.Drawing;


using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using OpenCvSharp;
using OpenCvSharp.Extensions;
using static System.Net.Mime.MediaTypeNames;

public class ImageEnhancement
{

    static Mat UnsharpMask(Mat image, double sigma, double strength, double threshold)
    {
        // 转换为灰度图
        Mat gray = image.Clone();
        //Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);

        // 应用高斯模糊
        Mat blurred = new Mat();
        Cv2.GaussianBlur(gray, blurred, new OpenCvSharp.Size(0, 0), sigma);

        // 计算锐化的差异
        Mat lowContrastMask = new Mat();
        Cv2.Absdiff(gray, blurred, lowContrastMask);

        // 设置阈值
        Mat mask = new Mat();
        Cv2.Threshold(lowContrastMask, mask, threshold, 255, ThresholdTypes.Binary);

        // 转换为浮点图像进行加权计算
        Mat floatImage = new Mat();
        image.ConvertTo(floatImage, MatType.CV_32F);
        Mat floatBlurred = new Mat();
        blurred.ConvertTo(floatBlurred, MatType.CV_32F);

        // 计算锐化后的图像
        Mat sharpened = new Mat();
        Cv2.AddWeighted(floatImage, 1.0 + strength, floatBlurred, -strength, 0, sharpened);

        // 转换回8位图像
        sharpened.ConvertTo(sharpened, MatType.CV_8U);

        return sharpened;
    }


    // 计算gamma值函数
    public static double GenGamma(Mat image)
    {
        // 将图像转换为灰度图像
        Mat grayImage = new Mat();
        Cv2.CvtColor(image, grayImage, ColorConversionCodes.BGR2GRAY);

        // 计算图像的均值
        Rect roi = new Rect(0, grayImage.Rows - 40, grayImage.Width, 40);//(x, y, width, height)
        Scalar meanval = Cv2.Mean(grayImage.SubMat(roi));

        double mean = meanval.Val0;

        double gamma;

        if (mean > 69)
        {
            gamma = 2.2;
        }
        else if (mean < 28)
        {
            gamma = 1.0;
        }
        else
        {
            gamma = (mean - 28) * 0.8 / 41.0 + 1.4;
        }

        Console.WriteLine($"gamma={gamma}");

        return gamma;
    }
    //gamma校正
    static Mat AdjustGamma(Mat image, double gamma)
    {
        Mat result = new Mat();
        Mat lut = new Mat(1, 256, MatType.CV_8UC1);

        double invGamma = gamma;
        for (int i = 0; i < 256; i++)
        {
            lut.Set(0, i, (byte)(Math.Pow(i / 255.0, invGamma) * 255.0));
        }

        Cv2.LUT(image, lut, result);
        return result;
    }


    // 图像增强方法
    public static Mat EnhanceMethod(Mat image)
    {
        double gamma = GenGamma(image);

        //// 对图像进行伽马校正


        Mat corrected_image = AdjustGamma(image, gamma);
        // 应用USM锐化
        double sigma = 1.0; // 高斯模糊的标准差
        double strength = 4; // 锐化强度
        double threshold = 0; // 阈值
        Mat USM_image = UnsharpMask(corrected_image, sigma, strength, threshold);
        Mat rst_image = new Mat();

        if (gamma == 1.0)
        {
            //corrected_image = image;
            rst_image = image;
        }
        else if (gamma < 2.0)
        {
            //corrected_image = 0.5 * corrected_image + corrected_image;
            //corrected_image.ConvertTo(corrected_image, MatType.CV_8U);

            Mat tmp_image = 0.5 * USM_image + USM_image;
            tmp_image.ConvertTo(rst_image, MatType.CV_8U);
        }
        else
        {
            rst_image = USM_image;
        }
        return rst_image;
    }
}

namespace DLLLibrary
{
    public class Class1
    {
        public Bitmap GImgCallback(Bitmap bitmapf)
        {

            //Mat img = Cv2.ImRead(bitmapf);
            // 将 Bitmap 转换为 Mat 对象
            Mat img = bitmapf.ToMat();
            if (img == null)
                return null;
            Mat enhancedImage = ImageEnhancement.EnhanceMethod(img);
            Bitmap pro = enhancedImage.ToBitmap();
            return pro;
        }



    }
}
