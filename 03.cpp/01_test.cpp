#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using namespace std;

class Student {
private:
    string name;
    int age;
    int id;
    vector<double> grades;

public:
    void input() {
        string line;
        getline(cin, line);
        stringstream ss(line);
        string token;

        // 读取姓名
        getline(ss, token, ',');
        name = token;

        // 读取年龄
        getline(ss, token, ',');
        age = stoi(token);

        // 读取学号
        getline(ss, token, ',');
        id = stoi(token);

        // 读取四个学年的成绩
        for (int i = 0; i < 4; ++i) {
            getline(ss, token, ',');
            double grade = stod(token);
            if (grade < 0) {
                cout << "Invalid grade: " << grade << endl;
                exit(1);
            }
            grades.push_back(grade);
        }
    }

    void calculate() {
        double sum = 0;
        for (double grade : grades) {
            sum += grade;
        }
        double average = sum / grades.size();
        grades.clear();
        grades.push_back(average);
    }

    void output() {
        cout << name << "," << age << "," << id << "," << grades[0] << endl;
    }
};

int main() {
    Student student;        // 定义类的对象
    student.input();        // 输入数据
    student.calculate();    // 计算平均成绩
    student.output();       // 输出数据
    return 0;
}