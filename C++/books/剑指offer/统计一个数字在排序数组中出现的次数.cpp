/*
统计一个数字在排序数组中出现的次数。
*/
#include <iostream>
#include <vector>
using namespace std;

class Solution{
public:
    int GetFirstK(vector<int> data, int k, int start, int end){
        // 二分找到第一个K
        int mid = (start + end) / 2;
        if(data[mid] == k){
            // 向左找第一个K
            if((mid > 0 && data[mid - 1] != k) || mid == 0)
                return mid;
            else
                end = mid - 1;
        }
        else if(data[mid] > k)
            end = mid - 1;
        else
            start = mid + 1;
        return GetFirstK(data, k, start, end);  
    }
    int GetLastK(vector<int> data, int k, int start, int end){
        // 二分找到最后一个个K
        int mid = (start + end) / 2;
        if(data[mid] == k){
            // 向右找第一个K
            if((mid < end && data[mid + 1] != k) || mid >= end)
                return mid;
            else
                start = mid + 1;
        }
        else if(data[mid] > k)
            end = mid - 1;
        else
            start = mid + 1;
        return GetLastK(data, k, start, end); 
    }
    int GetNumberOfK(vector<int> data, int k){
        // 核心思路：二分法找到第一个/最后一个k，然后计算差值，2*O(logn)
        // NOTE：二分法找到第一个k，然后前后遍历最差可能为O(n)
        int first = GetFirstK(data, k, 0, data.size() - 1);
        int last = GetLastK(data, k, 0, data.size() - 1);
        return last - first + 1;
    }
};

int main(){
    Solution s;
    vector<int> data = {1, 2, 3, 3, 3, 3, 4, 5};
    int k = 3;
    cout << s.GetNumberOfK(data, k) << endl;
    return 0;
}