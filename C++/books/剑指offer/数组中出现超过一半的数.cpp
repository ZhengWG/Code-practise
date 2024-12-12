/*
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
*/
#include <iostream>
#include <vector>

using namespace std;

class Solution{
public:
    int partition(vector<int> &numbers, int start, int end){
        // 基础的partition逻辑
        int pivot = numbers[start]; // TODO：随机选择pivot
        int i = start + 1;
        int j = end;
        while(i < j){
            while(i < j && numbers[j] >= pivot) j--;
            while(i < j && numbers[i] <= pivot) i++;
            if(i < j){
                swap(numbers[i], numbers[j]);
            }
        }
        // 最后交换i/j与pivot
        swap(numbers[start], numbers[i]);
        return i;
    }

    int MoreThanHalfNum_Solution_base(vector<int> numbers) {
        // 思路一：基于partion逻辑实现
        int n = numbers.size();
        int start = 0;
        int end = n - 1;
        // 基于partition逻辑，找到快排结果
        int index = partition(numbers, start, end);
        int mid = (n - 1) / 2 + 1;
        while(index != mid){
            if(index > mid){
                end = index - 1;
                index = partition(numbers, start, end);
            }else{
                start = index + 1;
                index = partition(numbers, start, end);
            }
        }
        return numbers[index];
    }

    int MoreThanHalfNum_Solution_advanced(vector<int> numbers){
        // 思路二：基于摩尔投票法实现
        // 摩尔投票法的核心思想：
        // 1. 如果一个数字出现次数超过一半，那么该数字与其他所有数字相互抵消后，一定还能剩下
        // 2. 通过不断抵消不同的数字，最后剩下的数字就是可能的候选者
        
        if(numbers.empty()) return 0;
        
        int n = numbers.size();
        int num = numbers[0];  // 候选数字
        int cnt = 1;          // 候选数字的计数器
        
        // 第一步：找出候选数字
        for(int i = 1; i < n; i++){
            if(numbers[i] == num) {
                cnt++;        // 遇到相同数字，计数器加1
            } else {
                cnt--;        // 遇到不同数字，计数器减1
            }
            
            if(cnt == 0) {   // 当计数器为0时，更换候选数字
                num = numbers[i];
                cnt = 1;
            }
        }
        
        // 第二步：验证候选数字是否真的出现超过一半
        cnt = 0;
        for(int i = 0; i < n; i++){
            if(numbers[i] == num){
                cnt++;
            }
        }
        
        return cnt > n/2 ? num : 0;
    }
};


int main(){
    Solution sol;
    vector<int> numbers = {1,2,3,2,2,2,5,4,2};
    cout << sol.MoreThanHalfNum_Solution_base(numbers) << endl;
    cout << sol.MoreThanHalfNum_Solution_advanced(numbers) << endl;
    return 0;
}