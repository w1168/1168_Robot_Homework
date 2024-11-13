#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main() {
    // ����ͼ��ĳߴ�
    int width = 800;
    int height = 600;

    // ����һ���հ�ͼ�񣬱���Ϊ��ɫ
    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));

    // ����һ�� RotatedRect ����
    Point2f center(width / 2, height / 2);  // ���ĵ�
    Size2f size(200, 100);                  // ��С����Ⱥ͸߶ȣ�
    float angle = 315;                     // ��ת�Ƕȣ���λΪ�ȣ�

    RotatedRect rotatedRect(center, size, angle);

    // ������ת���ε��ĸ�����
    Point2f vertices[4];
    rotatedRect.points(vertices);

    // ��ͼ���ϻ�����ת����
    for (int i = 0; i < 4; ++i) {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
    }

    // ��ʾͼ��
    namedWindow("Rotated Rectangle", WINDOW_NORMAL);
    imshow("Rotated Rectangle", image);

    // �ȴ��û�����
    waitKey(0);

    return 0;
}