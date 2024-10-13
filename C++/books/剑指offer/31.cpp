/*
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。
ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
*/
class Solution {
public:
	//寻找每一位的含有1的规律
    int NumberOf1Between1AndN_Solution(int n)
    {
        if(n<=0)
            return 0;
        long long i = 1;
        int high = n/i;
        int low = n%i;
        int out = 0;
        while(high){
            int num = high%10;
            if(num<1)
                out += (high/10)*i;
            else if(num>1)
                out += (high/10+1)*i;
            else
                out += (high/10)*i+low+1;
            i *=10;
            high = n/i;
            low = n%i;
        }
        return out;
    }
};