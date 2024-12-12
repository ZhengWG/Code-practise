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
#include <iostream>
#include <vector>
using namespace std;

class Solution {
private:
    // 辅助数组，用于归并排序
    vector<int> temp;
    const int MOD = 1000000007;
    
    // 归并排序的合并操作
    long long mergeSort(vector<int>& data, int left, int right, vector<pair<int, int> >& result) {
        // 如果区间长度为1，则无需排序
        if (left >= right) return 0;
        
        int mid = left + (right - left) / 2;
        // 递归处理左右两半部分，并累加逆序对数量
        long long count = mergeSort(data, left, mid, result) + mergeSort(data, mid + 1, right, result);
        
        // 合并两个有序数组
        int i = left, j = mid + 1, k = left;
        
        // 将当前区间复制到临时数组
        for (int p = left; p <= right; p++) {
            temp[p] = data[p];
        }
        
        // 归并过程中统计逆序对
        while (i <= mid && j <= right) {
            if (temp[i] <= temp[j]) {
                // 非逆序数对，正常合并
                data[k++] = temp[i++];
            } else {
                // 当右半部分的数小于左半部分的数时，形成逆序对
                // 左半部分从i到mid的所有数都可以和temp[j]形成逆序对
                count = (count + (mid - i + 1)) % MOD;
                for(int p=i; p<=mid; p++){
                    result.push_back(make_pair(temp[p], temp[j]));
                }
                data[k++] = temp[j++];
            }
        }
        
        // 处理剩余元素
        while (i <= mid) {
            data[k++] = temp[i++];
        }
        while (j <= right) {
            data[k++] = temp[j++];
        }
        
        return count;
    }
    
public:
    vector<pair<int, int> > InversePairs(vector<int> data) {
        // 核心思路：基于归并排序实现
        // 为什么使用归并排序：归并排序在合并两个有序数组时，可以统计逆序对
        if (data.empty()) return vector<pair<int, int> >();
        
        // 初始化辅助数组
        temp.resize(data.size());
        vector<pair<int, int> > result;
        
        // 调用归并排序统计逆序对
        long long count = mergeSort(data, 0, data.size() - 1, result);
        return result;
    }
};

int main(){
    Solution sol;
    vector<int> data = {7, 5, 6, 4};
    vector<pair<int, int> > result = sol.InversePairs(data);
    for(auto it = result.begin(); it != result.end(); it++){
        cout << it->first << " " << it->second << endl;
    }
    return 0;
}