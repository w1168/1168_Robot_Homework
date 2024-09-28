//by 1168
//软件241WWY
#include<stdio.h>
//#include<stdlib.h>  //仅用来暂停

int main()
{
    
    int zs[5000];
    int N,i,j,k,x=1,jiashu;     //N为输入;i,j,k为循环控制;x为质数个数
    short flag=0;               //控制跳出,short省内存
    scanf("%d",&N);             //输入
    for (i=2;i<=N;i++)          //筛选所有质数
    {
        for (j=2;j<=i;j++)      //选定某个数开始验证是否为质数
        {
            if (j==i)
            {
                zs[x]=i;        //是质数就存进zs[]
                x++;
            }
            if (i%j==0)
            {
                break;          //不是质数就滚蛋
            }
            
        }
    }
    if (N%2==1)                 //处理N是奇数的情况
    {
        N--;
    }
    
    for(i=4;i<=N;i+=2)                                          //开始干正事(加数1+加数2=某个偶数)
    {
        flag=0;                                                 //i为偶数，zs[j]为加数1，jiashu为加数2
        for(j=1;j<=i;j++)                                       //先搞加数1
        {
            jiashu=i-zs[j];
            for(k=1;k<x;k++)                                    //再搞加数2
            {
                if(jiashu==zs[k])
                {
                    printf("%d=%d+%d\n",i,zs[j],jiashu);        //输出
                    flag=1;                                     //用flag来控制break防止多解
                    break;                                      //找到就跳出
                }
                
            }
            if (flag==1)                                        //找到一个解就跳出循环
                {
                    break;
                }
            
        }
    }
    //system("pause");
    
    
    
    /*
    for (i=1;i<x;i++)
    {
        printf("%d ",zs[i]);
    }
    */
    
    return 0;
}//normal