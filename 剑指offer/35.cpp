/*
在数组中的两个数字，如果前面一个数字大于后面的数字，
则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。
并将P对1000000007取模的结果输出。 即输出P%1000000007
输入描述:
题目保证输入的数组中没有的相同的数字

数据范围：

	对于%50的数据,size<=10^4

	对于%75的数据,size<=10^5

	对于%100的数据,size<=2*10^5
*/
class Solution {
public:
    long long Inversecore(vector<int> &data, vector<int> &copy, int start, int end){
        if(start==end){
            copy[start]=data[start];
            return 0;
        }
        int length = (end-start)/2;
        long long left = Inversecore(copy,data,start,start+length);
        long long right = Inversecore(copy,data,start+length+1,end);
        int i = start+length;
        int j = end;
        int incopy = end;
        long long count = 0;
        while(i>=start && j>=start+length+1){
            if(data[i]>data[j]){
                copy[incopy--]=data[i--];
                count += j - start - length;
                if(count > 1000000007)
                    count  = count % 1000000007;
            }
            else
                copy[incopy--]=data[j--];
        }
        for(;i>=start;i--)
            copy[incopy--]=data[i];
        for(;j>=start+length+1;j--)
            copy[incopy--]=data[j];
        return left+right+count;
    }
    int InversePairs(vector<int> data) {
        if(!data.size())
            return 0;
        vector<int> copy = data;
        int num = Inversecore(data,copy,0,data.size()-1)%1000000007;
        return num;
    }
};