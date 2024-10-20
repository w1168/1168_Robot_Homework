//by 1168
//蓝框为跟踪，红点为装甲板中心，黑色为预测点，绿色的线是预测指示线，线越长代表速度越快
//main函数里可以快速改颜色,默认是蓝色，因为蓝色效果比较好，红色有几帧实在没办法（哭了）
//处理一帧平均约为32ms(一开始是34ms，越运行平均耗时会越低，直到结束差不多是30ms)，平均值可以再终端里看
//最耗时的语句是canny，需要6到8ms，canny中的非极大值抑制需要对每个像素的梯度方向进行判断，并在梯度方向上进行比较，保留局部最大值。这个过程涉及复杂的条件判断和像素访问。
//其次耗时的语句是读取图片，也就是.read（），差不多要5ms

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
//#include "kalman.cpp"

using namespace cv;
using namespace std;



double getDistance(Point pointO, Point pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);
	return distance;
}

class predict		//自己研发的预测类，本质是取前三个点进行预测，两段距离用来预测速度，后一段预测方向
{
private:
	vector<Point> allframe_point;	//存所有的帧
	vector<int> perframe;
	int now = 0;		//现在的循环位置，向前倒3个来预测
	//int allfps;    //总帧数

public:
	void chushihua(int allfps)			//这里填入视频有多少帧
	{
		allframe_point = vector<Point>(allfps);
		perframe = vector<int>(allfps);
		now = 0;
	}
	
	void getpoint(Point point , int fps)
	{
		allframe_point[now] = point;
		perframe[now] = fps;
		now++;
	}
	Point predictpoint(Mat img)
	{
		float vx = 0,vy = 0;
		float a, b, c;		//a>x,b>y,c>斜边
		if (now > 2)
		{
			vx = ((allframe_point[now].x - allframe_point[now - 1].x) / (perframe[now] - perframe[now - 1]) + (allframe_point[now - 1].x - allframe_point[now - 2].x) / (perframe[now - 1] - perframe[now - 2])) / 2;
			vy = ((allframe_point[now].y - allframe_point[now - 1].y) / (perframe[now] - perframe[now - 1]) + (allframe_point[now - 1].y - allframe_point[now - 2].y) / (perframe[now - 1] - perframe[now - 2])) / 2;
			a = (allframe_point[now].x - allframe_point[now - 1].x);
			b = (allframe_point[now].y - allframe_point[now - 1].y);
			if (a < 0) {a = -a;}
			if (b < 0) { b = -b; }
			c = sqrtf(a * a + b * b);
			//cout << a << "  " << vy << "  "<< allframe_point[now - 1]<<" "<< allframe_point[now - 1].x + (a / c) * vx;
			if(getDistance(allframe_point[now - 1], Point(allframe_point[now - 1].x + (a / c) * vx * 10, allframe_point[now - 1].y + (b / c) * vy * 10)) < 50)
			{
				line(img, allframe_point[now - 1], Point(allframe_point[now - 1].x + (a / c) * vx * 10, allframe_point[now - 1].y + (b / c) * vy * 10), Scalar(0, 255, 0), 4);
			}
			return Point(allframe_point[now - 1].x + (a/c)*vx, allframe_point[now - 1].y + (b/c) * vy);
		
		}
	
	
	}


};


struct mainpoint
{
	Point mp;
	Rect rect1, rect2;
};
	



mainpoint Rectcompare(Rect rect1,Rect rect2,Mat img)  
{
	int point = 8;
	//加分  if中不能用连续比较符号
	//if (rect2.tl().y < (rect1.tl().y + rect1.br().y) / 2 && (rect1.tl().y + rect1.br().y) / 2 < rect2.br().y) { point+=1; }//中点对齐检测
	//cout << point << "/1  ";
	//if (rect1.tl().y < (rect2.tl().y + rect2.br().y) / 2 && (rect2.tl().y + rect2.br().y) / 2 < rect1.br().y) { point+=1; }
	//cout << point << "/2  ";
	//
	//if (100 < rect1.tl().x - rect2.tl().x && rect1.tl().x - rect2.tl().x < 400) { point+=2; }//间距检测
	//cout << point << "/4  ";
	//
	//if (20 < rect1.br().y - rect1.tl().y && rect1.br().y - rect1.tl().y < 100) { point+=1; }//数值检测
	//cout << point << "/5  ";
	//
	//if (20 < rect2.br().y - rect2.tl().y && rect2.br().y - rect2.tl().y < 100) { point+=1; }
	//cout << point << "/6  ";
	//if (2 < rect1.br().x - rect1.tl().x && rect1.br().x - rect1.tl().x < 30) { point+=1; }
	//cout << point << "/7  ";
	//if (2 < rect2.br().x - rect2.tl().x && rect2.br().x - rect2.tl().x < 30) { point+=1; }
	//cout << point << "/8  ";
	//cout << rect1.br().y - rect1.tl().y << "   ";

	//减分
	if ((rect2.br().y - rect2.tl().y) < (rect2.br().x - rect2.tl().x)) { point = 0; }//竖长方体检测
	if ((rect1.br().y - rect1.tl().y) < (rect1.br().x - rect1.tl().x)) { point = 0; }
	
	if (rect2.contains((rect1.br() + rect1.tl()) / 2)) { point = 0; } //重合检测
	if (rect1.contains((rect2.br() + rect2.tl()) / 2)) { point = 0; }
	if (rect1.contains( rect2.br() )) { point = 0; }
	if (rect1.contains( rect2.tl() )) { point = 0; }
	if (rect2.contains( rect1.br() )) { point = 0; }
	if (rect2.contains( rect1.tl() )) { point = 0; }

	
	//cout << rect1 << rect2 << point << endl;
	if (point == 8)
	{
		//rectangle(img, rect1.tl(), rect1.br(), Scalar(255, 255, 0), 5);
		//rectangle(img, rect2.tl(), rect2.br(), Scalar(255, 255, 0), 5);
		return { (rect1.br() + rect2.tl()) / 2,rect1,rect2 };
	}
	else { return { Point(0, 0),rect1,rect2 }; }
}

Mat getmask(Mat img , int borr)									//颜色检测
{
	//蓝:70,179,30,255,245,255
	//红:0 60 40 128 240 255
	Mat mask, img_hsv;
	int hmin = 70, smin = 30, vmin = 245;				//设置颜色识别范围
	int hmax = 179, smax = 255, vmax = 255;				//这是蓝的

	if (borr == 1)
	{
		hmin = 0, smin = 40, vmin = 240;				//这是红的
		hmax = 60, smax = 128, vmax = 255;
	
	}
	

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

vector<Rect> getallRect(Mat img_dil ,int *pa,Mat img)								//这里的img_dil要预处理,img不用
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
		if (area > 150 && area < 2500)									//改分辨率这里要×4
		{
			float peri = arcLength(contours[i], true);					//设置一个检测精度
			approxPolyDP(contours[i], conpoly[i], 0.2 * peri, true);
			boundRect[a] = boundingRect(conpoly[i]);
			//drawContours(img, conpoly, -1, Scalar(100, 100, 255), 1);
			//rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 1);
			a++;
		}
	}
	*pa = a;													//将vector长度传递出去,最后多加了一次所以后面只能<不能<=否则会下标越界

	return boundRect;
}



void main()
{
	//Kalman kf(1.0, 0.1, 0.1);
	//Mat z = (cv::Mat_<double>(2, 1) << 0,0);
	
	//=========================此处改颜色！！==========================================================
	int red_or_blue = 2;					//1为红，2为蓝                                          ===
	//=================================================================================================
	string path;
	if (red_or_blue == 1) { path = "1.avi"; }
	if (red_or_blue == 2) { path = "2.avi"; }//原视频是 1440*1080 60hz 
	VideoCapture cap(path);										//捕捉视频
	int frame_count = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));	//获取帧数
	Mat img,mask,img_pre;
	int fps = 0;
	float timeadd = 0;
	mainpoint lastpoint;

	predict pred;
	pred.chushihua(frame_count);
	
	while (true)
	{
		
		float mt;
		int t1, t2;
		int e1 = getTickCount();
		
		int a = 0, * pa = &a;
		vector<Rect> boundRect;
		mainpoint mpoint;
		cap.read(img);											//视频读取成图片
		resize(img, img, Size(), 0.5, 0.5);						//这里缩放了0.5，如果要回去了话，前面的compare要*2 area*4
		//resize(img,img,Size(960, 720),0,0,INTER_LINEAR);
		mask = getmask(img,red_or_blue);
		img_pre = preprocess(mask);
		boundRect = getallRect(img_pre, pa, img);
		
		
	
		vector<mainpoint> allpoint( 1000 );
		Point nextpoint;
		int i1 = 0, i2 = 0, mun = 0;
		for (i1 = 0; i1 < a; i1++)
		{
			for (i2 = i1 + 1; i2 < a; i2++)
			{

				if (Rectcompare(boundRect[i1], boundRect[i2], img).mp != Point(0, 0))
				{
					mpoint = Rectcompare(boundRect[i1], boundRect[i2], img);
					//circle(img, mpoint.mp, 1, Scalar(255, 255, 255), -1);    //画出所有检测到的点，白色
					allpoint[mun] = mpoint;
					mun++;
				}

			}
		}
		
		
		
		//用上一帧的正方体来检测
		
		int themun = 0;
		if (fps <= 2 || mun == 1)
		{
			lastpoint = mpoint;
			pred.getpoint(lastpoint.mp,fps);
		}
		
		if (fps != 0 && mun != 1)
		{
			
			vector<float> dis(mun + 1);
			for (i1 = 0; i1 < mun; i1++)
			{
				dis[i1] = getDistance( lastpoint.rect1.tl() , lastpoint.rect2.tl() ) - getDistance( allpoint[i1].rect1.tl(), allpoint[i1].rect2.tl() );
				if (dis[i1] < 0) { dis[i1] = -dis[i1]; }
			}
			
			int min_value = dis[0]; //找距离的差最小的那个
			
			for (i1 = 1; i1 < mun; i1++) 
			{
				if (dis[i1] < min_value) 
				{
					min_value = dis[i1]; 
					themun = i1;
				}
			}
			lastpoint = allpoint[themun];
			pred.getpoint(lastpoint.mp, fps);
		}
		rectangle(img,allpoint[themun].rect1.tl(), allpoint[themun].rect1.br(), Scalar(255, 255, 0), 5);
		rectangle(img, allpoint[themun].rect2.tl(), allpoint[themun].rect2.br(), Scalar(255, 255, 0), 5);
		circle(img, allpoint[themun].mp, 10, Scalar(0, 0, 255), -1);
		
		
		
		
		//预测部分
		if (fps > 2)
		{
			circle(img, pred.predictpoint(img), 5, Scalar(0, 0, 0), -1);
			//cout << pred.predictpoint() << "....." << lastpoint.mp << endl;
		}
		
		imshow("img", img);
		fps++;
		waitKey(1);
		
		//计时部分
		int e2 = getTickCount();
		mt = ((e2 - e1) / getTickFrequency()) * 1000;
		timeadd += mt;
		cout << "此帧用时："<< mt << "ms";
		cout << "  平均用时："<< timeadd / fps <<" fps:" << fps << endl;
	}
}

