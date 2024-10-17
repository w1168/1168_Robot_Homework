#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat getmask(Mat img)												//��ɫ���
{
	//��:60,179,0,255,245,255
	//��
	Mat mask, img_hsv;
	int hmin = 0, smin = 110, vmin = 153;				//������ɫʶ��Χ
	int hmax = 19, smax = 240, vmax = 255;
	
	cvtColor(img, img_hsv,COLOR_BGR2HSV);				//ת����ɫ
	
	Scalar lower(hmin, smin, vmin);
	Scalar upper(hmax, smax, vmax);

	inRange(img_hsv, lower, upper, mask);				//��ȡͼ��,�����ɰ�
	
	return mask;
	
}



Mat preprocess(Mat img)
{
	Mat img_gray, img_blur, img_canny, img_dilate;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	GaussianBlur(img_gray, img_blur, Size(3, 3), 5, 0);
	Canny(img_blur, img_canny, 50, 150);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(img_canny, img_dilate, kernel);											//�������ñ�Ե�պ�
	return img_dilate;
}





void main()
{
	string path = "Resources/test_video.mp4";
	VideoCapture cap(path);							//��׽��Ƶ
	Mat img,mask,img_pre;
	while (true) {
		cap.read(img);								//��Ƶ��ȡ��ͼƬ
		
		mask = getmask(img);
		img_pre = preprocess(mask);
		
		
		
		
		
		
		
		waitKey(1);

	}
}