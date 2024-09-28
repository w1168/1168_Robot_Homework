//by 1168
//第2个作业，排序
//输入最后没有空格，中间有空格
#include <stdio.h>

int main()
{   
    int a[100000],n=0,i=0,j=0;          //用最大的数字，最大10^9<int的极限,n是N,ij是循环变量
    scanf("%d",&n);
    for (i=0; i<n-1; i++)
    {
        scanf("%d",&a[i]);              //读取数字
        getchar();                      //处理掉空格
    }
    i++;
    scanf("%d",&a[i]);                  //由于最后没有空格，所以得单独拉出来
    
    for(i=0;i<=n-1;i++)                 //标准の冒泡排序
    {
        for(j=i+1;j<n;j++)
        {
            if(a[i]>a[j])
            {
                a[i]=a[i]+a[j];         //无中继交换
                a[j]=a[i]-a[j];
                a[i]=a[i]-a[j];
            }
        }
    }
   
    for(i=0;i<10;i++)                   //输出部分
    {
        if(a[i]!=0)                     //后面的0别输出
        {
            printf("%d ",a[i]);         //细节小空格
        }
    }
    
    return 0;
}//EZ
