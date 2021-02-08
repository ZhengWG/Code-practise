/*
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
*/
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if(!numbers.size())
            return 0;
        int num = numbers[0];
        int cnt = 1;
		//若果有这样的数，必然满足下面条件
        for(int i=1;i<numbers.size();i++){
            if(numbers[i]==num)
                cnt++;
            else
                cnt--;
            if(!cnt){
                cnt = 1;
                num = numbers[i];
            }
        }
        cnt = 0;
        for(int i=0;i<numbers.size();i++)
            if(numbers[i]==num)
                cnt++;
        if(cnt*2>numbers.size())
            return num;
        return 0;
    }
};