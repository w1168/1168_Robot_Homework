#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;								//���˺�.h��������
using namespace cv;


void main()
{
	string path = "2.avi";
	VideoCapture cap(path);							//��׽��Ƶ

	Mat img;
	
	cap.set(CAP_PROP_POS_FRAMES, 450);				//����Ҫ��ȡ��֡��
	cap.read(img);								//read��������һ������ֵ��һ����Ƶ֡����֡��ȡ�ɹ����򷵻�True
	
	
	
	
									//��Ƶ��ȡ��ͼƬ

		Mat mask, img_hsv;

		int hmin = 0, smin = 10, vmin = 245;				//������ɫʶ��Χ
		int hmax = 60, smax = 128, vmax = 255;


		//cvtColor(img, img_hsv, COLOR_BGR2HSV);				//ת����ɫ

		namedWindow("Trackbars", (640, 200));					//����trackbars����
		
		createTrackbar("hue min", "Trackbars", &hmin, 179);	//Ѱ����ɫ��Χ
		createTrackbar("hue max", "Trackbars", &hmax, 179);
		createTrackbar("sat min", "Trackbars", &smin, 255);
		createTrackbar("sat max", "Trackbars", &smax, 255);
		createTrackbar("val min", "Trackbars", &vmin, 255);
		createTrackbar("val max", "Trackbars", &vmax, 255);

		while (true)
		{
			//cap.read(img);	
		
			cvtColor(img, img_hsv, COLOR_BGR2HSV);				//ת����ɫ
			Scalar lower(hmin, smin, vmin);
			Scalar upper(hmax, smax, vmax);

			inRange(img_hsv, lower, upper, mask);				//��ȡͼ��,�����ɰ�
			
			namedWindow("Image", 0);
			resizeWindow("Image", 640, 480);
			namedWindow("mask", 0);
			resizeWindow("mask", 640, 480);
			imshow("Image", img);
			//imshow("image hsc", img_hsv);
			imshow("mask", mask);
			waitKey(1);
		}	
		
		




		//imshow("Image", img);						//չʾÿһ��ͼƬ

		waitKey(0);

	
}

