//by 1168
//����Ϊ���٣����Ϊװ�װ����ģ���ɫΪԤ��㣬��ɫ������Ԥ��ָʾ�ߣ���Խ�������ٶ�Խ��
//main��������Կ��ٸ���ɫ,Ĭ������ɫ����Ϊ��ɫЧ���ȽϺã���ɫ�м�֡ʵ��û�취�����ˣ�
//����һ֡ƽ��ԼΪ32ms(һ��ʼ��36ms��Խ����ƽ����ʱ��Խ�ͣ�ֱ�����������32ms)��ƽ��ֵ�������ն��￴
//���ʱ�������canny����Ҫ6��8ms��canny�еķǼ���ֵ������Ҫ��ÿ�����ص��ݶȷ�������жϣ������ݶȷ����Ͻ��бȽϣ������ֲ����ֵ����������漰���ӵ������жϺ����ط��ʡ�
//��κ�ʱ������Ƕ�ȡͼƬ��Ҳ����.read���������Ҫ5ms

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

struct mainpoint
{
	Point mp;
	RotatedRect rect1, rect2;
};

double getDistance(Point pointO, Point pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);
	return distance;
}

class predict		//�Լ��з���Ԥ���࣬������ȡǰ���������Ԥ�⣬���ξ�������Ԥ���ٶȣ���һ��Ԥ�ⷽ��
{
private:
	vector<Point> allframe_point;	//�����е�֡
	vector<int> perframe;
	int now = 0;		//���ڵ�ѭ��λ�ã���ǰ��3����Ԥ��
	//int allfps;    //��֡��

public:
	void chushihua(int allfps)			//����������Ƶ�ж���֡
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
		float a, b, c;		//a>x,b>y,c>б��
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

mainpoint Rectcompare(RotatedRect rect1, RotatedRect rect2,Mat img)    //�����ȶ�
{
	/*
	rect1.center : ���ε����ĵ㣬����Ϊ cv::Point2f��
	rect1.size : ���εĴ�С����Ⱥ͸߶ȣ�,����Ϊ cv::Size2f(width,height)��
	rect1.angle : ���ε���ת�Ƕȣ���λΪ�ȣ���ʱ��,��Χ��[0, 360?)��
	*/
	
	int point = 8;
    //����
	//int min = 1, max = 200;
	//if (rect2.size.height > rect2.size.width)//����������
	//{
	//	if (rect2.size.height / rect2.size.width < min || rect2.size.height / rect2.size.width > max) { point = 0; }
	//}
	//else if (rect2.size.height < rect2.size.width)
	//{
	//	if (rect2.size.width / rect2.size.height < min || rect2.size.width / rect2.size.height > max) { point = 0; }

	//}
	//else {point = 0;}
	//if (rect1.size.height > rect1.size.width)
	//{
	//	if (rect1.size.height / rect1.size.width < min || rect1.size.height / rect1.size.width > max) { point = 0; }
	//}
	//else if (rect1.size.height < rect1.size.width)
	//{
	//	if (rect1.size.width / rect1.size.height < min || rect1.size.width / rect1.size.height > max) { point = 0; }

	//}
	//else { point = 0; }
	
	if (rect2.center == rect1.center) { point = 0; } //�Լ����Լ���⣨��Ч��

	Rect straight_rect1, straight_rect2;
	straight_rect1 = rect1.boundingRect();
	straight_rect1 = rect1.boundingRect();
	if (straight_rect2.contains((straight_rect1.br() + straight_rect1.tl()) / 2)) { point = 0; } //�غϼ��
	if (straight_rect1.contains((straight_rect2.br() + straight_rect2.tl()) / 2)) { point = 0; }
	if (straight_rect1.contains(straight_rect2.br() )) { point = 0; }
	if (straight_rect1.contains(straight_rect2.tl() )) { point = 0; }
	if (straight_rect2.contains(straight_rect1.br() )) { point = 0; }
	if (straight_rect2.contains(straight_rect1.tl() )) { point = 0; }
	
	//cout << rect1 << rect2 << point << endl;
	if (point == 8)
	{
		return { (rect1.center + rect2.center) / 2,rect1,rect2};
	}
	else { return { Point(0, 0),rect1,rect2 }; }
}

Mat getmask(Mat img , int borr)									//��ɫ���
{
	//��:70,179,30,255,245,255
	//��:0 60 40 128 240 255
	Mat mask, img_hsv;
	int hmin = 70, smin = 30, vmin = 245;				//������ɫʶ��Χ
	int hmax = 179, smax = 255, vmax = 255;				//��������

	if (borr == 1)
	{
		hmin = 0, smin = 40, vmin = 240;				//���Ǻ��
		hmax = 60, smax = 128, vmax = 255;
	
	}
	

	cvtColor(img, img_hsv,COLOR_BGR2HSV);				//ת����ɫ
	
	Scalar lower(hmin, smin, vmin);
	Scalar upper(hmax, smax, vmax);

	inRange(img_hsv, lower, upper, mask);				//��ȡͼ��,�����ɰ�
	
	return mask;
	
}

Mat preprocess(Mat img)
{
	Mat img_blur, img_canny, img_dilate;
	GaussianBlur(img, img_blur, Size(3, 3), 5, 0);
	Canny(img_blur, img_canny, 50, 150);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(img_canny, img_dilate, kernel);											//�������ñ�Ե�պ�
	return img_dilate;
}

vector<RotatedRect> getallRect(Mat img_dil ,int *pa,Mat img)								//�����img_dilҪԤ����,img����
{
	vector<vector<Point>> contours;													//Ƕ������,vector�ܴ���point��vector��vector������������������
	vector<Vec4i> hierarchy;
	findContours(img_dil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//�ұ�Ե
	
	int a = 0 ;
	vector<RotatedRect> boundRect(contours.size());
	//vector<vector<Point>> conpoly(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);							//�����ͼ�����
		if (area > 150 && area < 2500)									//�ķֱ�������Ҫ��4
		{
			//float peri = arcLength(contours[i], true);					//����һ����⾫��
			//approxPolyDP(contours[i], conpoly[i], 0.2 * peri, true);	//Ѱ�ұպ�����
			boundRect[a] = minAreaRect(contours[i]);						//Ѱ����С����
			a++;
		}
	}
	*pa = a;													//��vector���ȴ��ݳ�ȥ,�������һ�����Ժ���ֻ��<����<=������±�Խ��
	return boundRect;
}



void main()
{
	//=========================�˴�����ɫ����==========================================================
	int red_or_blue = 1;					//1Ϊ�죬2Ϊ��                                          ===
	//=================================================================================================
	string path;
	if (red_or_blue == 1) { path = "1.avi"; }
	if (red_or_blue == 2) { path = "2.avi"; }//ԭ��Ƶ�� 1440*1080 60hz 
	VideoCapture cap(path);										//��׽��Ƶ
	int frame_count = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));	//��ȡ֡��
	Mat img,mask,img_pre;
	int fps = 0;
	float timeadd = 0;
	mainpoint lastpoint;

	//��ʼ��Ԥ����
	predict pred;
	pred.chushihua(frame_count);
	
	//�����￪ʼ����һ֡
	while (true)
	{
		//�����������
		float mt;
		int t1, t2;
		int e1 = getTickCount();
		int a = 0, * pa = &a;
		vector<RotatedRect> boundRect;
		mainpoint mpoint;
		
		
		//������
		cap.read(img);//��ȡ									//��Ƶ��ȡ��ͼƬ
		resize(img, img, Size(), 0.5, 0.5);//����				//����������0.5�����Ҫ��ȥ�˻���ǰ���compareҪ*2 area*4
		mask = getmask(img,red_or_blue);//�ɰ�
		img_pre = preprocess(mask);//Ԥ����
		boundRect = getallRect(img_pre, pa, img);//��ȡ���о��ο�
		
		
		//�ȶԲ���
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
					//circle(img, mpoint.mp, 1, Scalar(255, 255, 255), -1);    //�������м�⵽�ĵ㣬��ɫ
					allpoint[mun] = mpoint;
					mun++;
				}

			}
		}
		
		
		
		//����һ֡�������������
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
				dis[i1] = getDistance( lastpoint.rect1.center , lastpoint.rect2.center ) - getDistance( allpoint[i1].rect1.center, allpoint[i1].rect2.center );
				if (dis[i1] < 0) { dis[i1] = -dis[i1]; }
			}
			
			int min_value = dis[0]; //�Ҿ���Ĳ���С���Ǹ�
			
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
		
		
		//���Ʋ���
		// ������ת���ε��ĸ�����
		Point2f vertices[4];
		allpoint[themun].rect1.points(vertices);//��һ��
		// ��ͼ���ϻ�����ת����
		for (int r1 = 0; r1 < 4; r1++) {
			line(img, vertices[r1 ], vertices[(r1 + 1) % 4], Scalar(255, 0, 255), 2);
		}
		allpoint[themun].rect2.points(vertices);//�ڶ���
		// ��ͼ���ϻ�����ת����
		for (int r2 = 0; r2 < 4; r2++) {
			line(img, vertices[r2], vertices[(r2 + 1) % 4], Scalar(255, 0, 255), 2);
		}
		circle(img, allpoint[themun].mp, 8, Scalar(0, 0, 255), -1);
		
		
		//Ԥ�ⲿ��
		if (fps > 2)
		{
			circle(img, pred.predictpoint(img), 5, Scalar(0, 0, 0), -1);
		}
		imshow("img", img);
		fps++;
		waitKey(1);
		
		
		//��ʱ����
		int e2 = getTickCount();
		mt = ((e2 - e1) / getTickFrequency()) * 1000;
		timeadd += mt;
		cout << "��֡��ʱ��"<< mt << "ms";
		cout << "  ƽ����ʱ��"<< timeadd / fps <<" fps:" << fps << endl;
	}
}




