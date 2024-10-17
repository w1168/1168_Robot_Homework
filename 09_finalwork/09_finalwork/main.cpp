#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


double getDistance(Point pointO, Point pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);
	return distance;
}


Point Rectcompare(Rect rect1,Rect rect2,Mat img)
{
	int point = 0;
	//加分
	if (rect2.tl().y < (rect1.tl().y + rect1.br().y) / 2 < rect2.br().y) { point++; }//中点对齐检测
	if (rect1.tl().y < (rect2.tl().y + rect2.br().y) / 2 < rect1.br().y) { point++; }

	if (120 < rect1.tl().x - rect2.tl().x < 240) { point++; }//间距检测


	//减分
	if ((rect2.br().y - rect2.tl().y) < (rect2.br().x - rect2.tl().x)) { point = 0; }//竖长方体检测
	if ((rect1.br().y - rect1.tl().y) < (rect1.br().x - rect1.tl().x)) { point = 0; }
	
	if (rect2.contains((rect1.br() + rect1.tl()) / 2)) { point = 0; } //重合检测
	if (rect1.contains((rect2.br() + rect2.tl()) / 2)) { point = 0; }
	if (rect1.contains( rect2.br() )) { point = 0; }
	if (rect1.contains( rect2.tl() )) { point = 0; }
	if (rect2.contains( rect1.br() )) { point = 0; }
	if (rect2.contains( rect1.tl() )) { point = 0; }

	if (point > 2)
	{
		rectangle(img, rect1.tl(), rect1.br(), Scalar(255, 255, 0), 5);
		rectangle(img, rect2.tl(), rect2.br(), Scalar(255, 255, 0), 5);
		return (rect1.br() + rect2.tl()) / 2;
	}
	else { return Point(0, 0); }
}

Mat getmask(Mat img)									//颜色检测
{
	//蓝:70,179,0,255,245,255
	//红
	Mat mask, img_hsv;
	int hmin = 70, smin = 0, vmin = 245;				//设置颜色识别范围
	int hmax = 179, smax = 255, vmax = 255;
	
	cvtColor(img, img_hsv,COLOR_BGR2HSV);				//转换颜色
	
	Scalar lower(hmin, smin, vmin);
	Scalar upper(hmax, smax, vmax);

	inRange(img_hsv, lower, upper, mask);				//读取图像,生成蒙版
	
	return mask;
	
}



Mat preprocess(Mat img)
{
	Mat img_blur, img_canny, img_dilate;
	GaussianBlur(img, img_blur, Size(3, 3), 5, 0);
	Canny(img_blur, img_canny, 50, 150);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(img_canny, img_dilate, kernel);											//膨胀能让边缘闭合
	return img_dilate;
}

vector<Rect> getallRect(Mat img_dil ,int *pa,Mat img)												//这里的img_dil要预处理,img不用
{
	vector<vector<Point>> contours;													//嵌套容器,vector能存多个point，vector存vector能做到存多个这样的组
	vector<Vec4i> hierarchy;
	findContours(img_dil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//找边缘
	//drawContours(img,contours,-1,Scalar(255,0,255),2);

	int a = 0 ;
	vector<Rect> boundRect(contours.size());
	vector<vector<Point>> conpoly(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);							//检测封闭图形面积
		if (area > 200 && area < 8000)									//改分辨率这里要×4
		{
			float peri = arcLength(contours[i], true);					//设置一个检测精度
			approxPolyDP(contours[i], conpoly[i], 0.02 * peri, true);
			boundRect[a] = boundingRect(conpoly[i]);
			drawContours(img, conpoly, -1, Scalar(100, 100, 255), 1);
			rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2);
			a++;
		}
	}
	*pa = a;													//将vector长度传递出去,最后多加了一次所以后面只能<不能<=否则会下标越界

	return boundRect;
}



void main()
{
	string path = "2.avi";											//原视频是 1440*1080 60hz > 960 720
	VideoCapture cap(path);										//捕捉视频
	Mat img,mask,img_pre;
	while (true)
	{
		int a = 0, * pa = &a;
		vector<Rect> boundRect;
		Point mainpoint;
		cap.read(img);											//视频读取成图片

		resize(img, img, Size(), 0.5, 0.5);						//这里缩放了0.5，如果要回去了话，前面的compare要*2
		//resize(img,img,Size(960, 720),0,0,INTER_LINEAR);
		mask = getmask(img);
		img_pre = preprocess(mask);
		boundRect = getallRect(img_pre,pa,img);
		

		int i1 = 0, i2 = 0;
		
		for (i1 = 0; i1 < a; i1++)
		{
			for (i2 = i1 + 1; i2 < a; i2++)
			{

				if (Rectcompare(boundRect[i1], boundRect[i2],img) != Point(0, 0))
				{
					mainpoint = Rectcompare(boundRect[i1], boundRect[i2], img);
					circle(img, mainpoint, 10, Scalar(255,255,255), -1);

				}

			}
		}
		imshow("img", img);
		waitKey(1);
	}
}
		
