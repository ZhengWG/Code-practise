/*
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
*/
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class Solution{
public:
    string PrintMinNumber(vector<int> numbers) {
        // 核心思路：将数字转换为字符串，然后进行排序，使用字符串还有个好处，不会导致int溢出
        // 排序规则：两个数字拼接后的字符串字典序最小
        // 使用自定义排序函数
        sort(numbers.begin(), numbers.end(), [](int a, int b) {
            return to_string(a) + to_string(b) < to_string(b) + to_string(a);
        }); // lambda表达式
        string result;
        for(int num : numbers) {
            result += to_string(num);
        }
        return result;
    }
};

int main(){
    Solution sol;
    vector<int> numbers = {3, 32, 321};
    cout << sol.PrintMinNumber(numbers) << endl;
    return 0;
}