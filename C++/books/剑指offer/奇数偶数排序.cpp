/*
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
*/

# include <iostream>
# include <vector>
using namespace std;

class Solution {
public:
    void reOrderArray_base(vector<int> &array) {
        // native version: 时间复杂度O(n)/空间复杂度O(n)
        // 保证奇数和奇数，偶数和偶数之间的相对位置不变
        vector<int> out;

        // 遍历数组，判断奇数
        auto it = array.begin();
        for (; it!=array.end(); it++) {
            if((*it & 0x01) == 1)
                out.emplace_back(*it);
        }

        // 遍历数组，判断偶数
        it = array.begin();
        for(it = array.begin(); it!=array.end(); it++){
            if((*it & 0x01) != 1)
                out.emplace_back(*it);
        }

        array = out;
    }

    void reOrderArray_twoPointer(vector<int> &array) {
        // 双指针：时间复杂度O(n)/空间复杂度O(1)
        if(array.size() == 0)
            return;
        int left = 0, right = array.size() - 1;
        while(left < right){
            // 移动左指针，从左到右，找到第一个偶数
            while(left < right && (array[left] & 0x01) == 1)
                left++;
            // 移动右指针，从右到左，找到第一个奇数
            while(left < right && (array[right] & 0x01) == 0)
                right--;
            // 交换左右值
            if(left < right)
                swap(array[left], array[right]);
        }
    }   
};

int main(){
    Solution s;
    vector<int> array1 = {1,2,3,4,5,6,7,8,9,10};
    s.reOrderArray_base(array1);
    cout << "base version: " << endl;
    for(auto it = array1.begin(); it!=array1.end(); it++)
        cout << *it << " ";
    cout << endl;

    vector<int> array2 = {1,2,3,4,5,6,7,8,9,10};
    s.reOrderArray_twoPointer(array2);
    cout << "two pointer version: " << endl;
    for(auto it = array2.begin(); it!=array2.end(); it++)
        cout << *it << " ";
    cout << endl;
    return 0;
}