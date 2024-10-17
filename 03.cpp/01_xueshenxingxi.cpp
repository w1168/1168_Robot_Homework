/*
输入
输入数据为一行，包括：
姓名,年龄,学号,第一学年平均成绩,第二学年平均成绩,第三学年平均成绩,第四学年平均成绩。
其中姓名为由字母和空格组成的字符串（输入保证姓名不超过20个字符，并且空格不会出现在字符串两端），年龄、学号和学年平均成绩均为非负整数。信息之间用逗号隔开。
输出
输出一行数据，包括：
姓名,年龄,学号,四年平均成绩。
信息之间用逗号隔开。
*/

//Tom Hanks,18,7817,80,80,90,70

#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <cstdlib>
using namespace std;

class Student {
private:
    char name[20];
    int age,mun,grade[4],i;
    float avg;char a;
public:
    void input()
    {
        
        for (i = 0; i < 20; i++)
        {
            a = getchar();
            if (a == ',')
                break;
            else
                name[i] = a;
        }
        scanf("%d,%d,%d,%d,%d,%d",&age,&mun,&grade[0],&grade[1],&grade[2],&grade[3]);
   
    }
    void calculate()
    {
        avg = (grade[0]+grade[1]+grade[2]+grade[3]) / 4.0;
    }void output()
    {
        
        for (i = 0; i < 20; i++)
        {
            
                putchar(name[i]);
        }
        cout <<  "," << age << "," << mun << "," << avg << endl;
    }
};

int main() {
	Student student;        // 定义类的对象
	student.input();        // 输入数据
	student.calculate();    // 计算平均成绩
	student.output();       // 输出数据
}
