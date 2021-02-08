/*
一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。
请写程序找出这两个只出现一次的数字。
*/
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        int sum = 0;
        //先分组：某位为1说明两个数字在该位上异号
        for(int i=0;i<data.size();i++){
            sum = sum ^ data[i]; 
        }
        int index = 0;
        while(1){
            if(sum & 1)
                break;
            sum = sum >> 1;
            index++;
        }
		//分组后进行查找
        int n1=0,n2=0;
        for(int i=0; i<data.size();i++){
            int tmp = data[i];
            tmp = tmp >> index;
            if(tmp & 1)
                n1 = n1 ^ data[i];
            else
                n2 = n2 ^ data[i];
        }
        *num1 = n1;
        *num2 = n2;
    }
};