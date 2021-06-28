using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCVForUnity;
using OpenCVForUnityExample;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.ImgprocModule;
using Rect = OpenCVForUnity.CoreModule.Rect;
using System;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.UnityUtils;
using TensorFlowLite;
using System.IO;

public class FaceDetect : MonoBehaviour
{
    WebCamTextureToMatExample webCamTextureToMat;
    string faceXml_path;
    string ModelFile = "model.tflite";
    Mat gray;
    MatOfRect faceRect;
    CascadeClassifier classifier;
    Interpreter interpreter;
    List<string> labels = new List<string>() { "Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised" };

    // Start is called before the first frame update
    void Start()
    {
        webCamTextureToMat = transform.GetComponent<WebCamTextureToMatExample>();
        //faceXml_path = Application.streamingAssetsPath + "/lbpcascade_frontalface.xml";
        faceXml_path = Utils.getFilePath("haarcascade_frontalface_alt2.xml", true);
        gray = new Mat();
        faceRect = new MatOfRect();
        classifier = new CascadeClassifier(faceXml_path);
        if (classifier.empty())
        {
            Debug.Log("classifier load failed");
        }
        else
        {
            Debug.Log("load successful");
        }
        //create tflite interpreter
        string modelPath = Path.Combine(Application.streamingAssetsPath, ModelFile);
        var options = new InterpreterOptions();
        options.AddGpuDelegate();
        options.threads = SystemInfo.processorCount;
        interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
        interpreter.LogIOInfo();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void DetectFace()
    {
        Imgproc.cvtColor(webCamTextureToMat.rgbaMat, gray, Imgproc.COLOR_RGB2GRAY);
        classifier.detectMultiScale(gray, faceRect, 1.1d, 2, 2, new Size(20, 20), new Size());
        Rect[] rects = faceRect.toArray();
        //Debug.LogWarning("number of faces: "+rects.Length.ToString());
        //webCamTextureToMat.rgbaMat = img;
        for (int i = 0; i < rects.Length; i++)
        {
            Size dsize = new Size(48, 48); // 设置新图片的大小
            Mat cropped_img = new Mat(dsize, CvType.CV_8UC1);// 创建一个新的Mat（opencv的矩阵数据类型）
            Mat roi_gray = new Mat(gray, rects[i]);
            Imgproc.resize(roi_gray, cropped_img, dsize);
            float[] inputs = new float[48 * 48];
            float[] outputs = new float[7];
            for (int j = 0; j < cropped_img.rows(); j++)
            {
                for (int k = 0; k < cropped_img.cols(); k++)
                {
                    inputs[j * 48 + k] = (float)cropped_img.get(j, k)[0];
                }
            }
            // 传入输入数据
            interpreter.SetInputTensorData(0, inputs);
            // 执行！！！
            interpreter.Invoke();
            // 获取输出数据
            interpreter.GetOutputTensorData(0, outputs);
            int idx = getMaxIndex(outputs);
            Imgproc.putText(webCamTextureToMat.rgbaMat, labels[idx], new Point(rects[i].tl().x, rects[i].tl().y), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 255, 255), 2, Imgproc.LINE_AA);
            Imgproc.rectangle(webCamTextureToMat.rgbaMat, new Point(rects[i].x, rects[i].y), new Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), new Scalar(0, 255, 0, 255), 2);
        }
        //Imgproc.rectangle(webCamTextureToMat.rgbaMat, new Point(0, 0), new Point(30, 30), new Scalar(0, 255, 0, 255), 2);
    }
    private int getMaxIndex(float[] arr)
    {
        float Max = 0;
        int index = 0;
        for (int i = 0; i < arr.Length; i++)
        {
            if (arr[i] > Max)
            {
                Max = arr[i];
                index = i;
            }
        }
        //Log.e(TAG, "max result: " + Max);
        return index;
    }
}
