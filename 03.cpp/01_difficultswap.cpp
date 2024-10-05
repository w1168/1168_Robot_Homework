#include <iostream>
using namespace std;

void swap(int*&a,int*&b)//此处为要填的
{
	int * tmp = a;			//书上的标准错误，这里应该是指针，而不是变量
	a = b;					//fail_swap2
	b = tmp;				//理解：这里仅仅交换了指针的值，而不是指针的内容，到后续就又变回来了
}							//如果想在函数中操作函数外的值，就要通过指针指的地址修改指针内容
int main()
{
	int a = 3,b = 5;
	int * pa = & a;
	int * pb = & b;
	swap(pa,pb);
	cout << *pa << "," << *pb;
	return 0;
}