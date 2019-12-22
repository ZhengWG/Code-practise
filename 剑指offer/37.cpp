/*
统计一个数字在排序数组中出现的次数。
*/
class Solution {
public:
	//二分法找到第一个出现位置
    int GetNumberOfK(vector<int> data ,int k) {
        int f = GetFirstOfK(data,k,0,data.size()-1);
        int cnt = 0;
        if(f==-1)
            return cnt;
        else{
            for(int i=f;i<data.size();i++){
                if(data[i]==k)
                    cnt++;
                else
                    break;
            }
        }
        return cnt;
    }
    int GetFirstOfK(vector<int> data, int k, int start, int end){
        if(start>end)
            return -1;
        int mid = (start+end)/2;
        if(data[mid]==k){
            if((mid>0 && data[mid-1]!=k)||mid==0){
                return mid;
            }
            else
                end = mid - 1;
        }
        else if(data[mid]>k)
            end = mid-1;
        else
            start = mid+1;
        return GetFirstOfK(data,k,start,end);
    }
};