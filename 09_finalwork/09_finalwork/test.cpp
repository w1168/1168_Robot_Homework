#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main() {
    // 定义图像的尺寸
    int width = 800;
    int height = 600;

    // 创建一个空白图像，背景为白色
    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));

    // 定义一个 RotatedRect 对象
    Point2f center(width / 2, height / 2);  // 中心点
    Size2f size(200, 100);                  // 大小（宽度和高度）
    float angle = 315;                     // 旋转角度（单位为度）

    RotatedRect rotatedRect(center, size, angle);

    // 计算旋转矩形的四个顶点
    Point2f vertices[4];
    rotatedRect.points(vertices);

    // 在图像上绘制旋转矩形
    for (int i = 0; i < 4; ++i) {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
    }

    // 显示图像
    namedWindow("Rotated Rectangle", WINDOW_NORMAL);
    imshow("Rotated Rectangle", image);

    // 等待用户按键
    waitKey(0);

    return 0;
}