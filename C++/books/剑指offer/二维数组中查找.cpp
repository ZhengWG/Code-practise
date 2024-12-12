/*
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，
判断数组中是否含有该整数。
*/
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    bool Find(int target, vector< vector<int> > array) {
        int row = 0;
        int col = array[0].size() - 1;
        while (row < array.size() && col >= 0) {
            if (array[row][col] > target) {
                col--;
            }
            else if (array[row][col] < target) {
                row++;
            }
            else
                return true;
        }
        return false;
    }
};


int main(){
    vector< vector<int> > array;
    array.push_back(vector<int>({1,2,3,4,5,6}));
    array.push_back(vector<int>({10,20,30,40,50,60}));
    int target1 = 10;
    int target2 = 15;
    
    Solution s = Solution();
    bool out1 = s.Find(target1, array);
    bool out2 = s.Find(target2, array);

    cout << "out for " << target1 << " is " << out1 << endl;
    cout << "out for " << target2 << " is " << out2 << endl;
    return 0;
}