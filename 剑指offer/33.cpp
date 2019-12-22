/*
把只包含质因子2、3和5的数称作丑数（Ugly Number）。
例如6、8都是丑数，但14不是，因为它包含质因子7。 
习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
*/
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if(index<=0)
            return 0;
        int t2=0,t3=0,t5=0;
        vector<int> nums(index);
        nums[0]=1;
		//注意乘法应该在本身的基础上进行
        for(int i=1;i<index;i++){
            nums[i]=min(nums[t2]*2,min(nums[t3]*3,nums[t5]*5));
            if(nums[i]==nums[t2]*2)
                t2++;
            if(nums[i]==nums[t3]*3)
                t3++;
            if(nums[i]==nums[t5]*5)
                t5++;
        }
        return nums[index-1];
    }
};