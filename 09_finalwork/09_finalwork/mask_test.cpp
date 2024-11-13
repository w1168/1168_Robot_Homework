#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;								//多人和.h不建议用
using namespace cv;


void main()
{
	string path = "2.avi";
	VideoCapture cap(path);							//捕捉视频

	Mat img;
	
	cap.set(CAP_PROP_POS_FRAMES, 300);				//设置要获取的帧号
	cap.read(img);								//read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
	
	
	
	
									//视频读取成图片

		Mat mask, img_hsv;

		int hmin = 0, smin = 10, vmin = 245;				//设置颜色识别范围
		int hmax = 60, smax = 128, vmax = 255;


		//cvtColor(img, img_hsv, COLOR_BGR2HSV);				//转换颜色

		namedWindow("Trackbars", (640, 200));					//创建trackbars窗口
		
		createTrackbar("hue min", "Trackbars", &hmin, 179);	//寻找颜色范围
		createTrackbar("hue max", "Trackbars", &hmax, 179);
		createTrackbar("sat min", "Trackbars", &smin, 255);
		createTrackbar("sat max", "Trackbars", &smax, 255);
		createTrackbar("val min", "Trackbars", &vmin, 255);
		createTrackbar("val max", "Trackbars", &vmax, 255);

		while (true)
		{
			//cap.read(img);	
		
			cvtColor(img, img_hsv, COLOR_BGR2HSV);				//转换颜色
			Scalar lower(hmin, smin, vmin);
			Scalar upper(hmax, smax, vmax);

			inRange(img_hsv, lower, upper, mask);				//读取图像,生成蒙版
			
			namedWindow("Image", 0);
			resizeWindow("Image", 640, 480);
			namedWindow("mask", 0);
			resizeWindow("mask", 640, 480);
			imshow("Image", img);
			//imshow("image hsc", img_hsv);
			imshow("mask", mask);
			waitKey(1);
		}	
		
		




		//imshow("Image", img);						//展示每一个图片

		waitKey(0);

	
}

