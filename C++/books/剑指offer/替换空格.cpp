/*
题目：请实现一个函数，把字符串中的每个空格替换成"%20"。例如输入“We arehappy.”​，则输出“We%20are%20happy.”​。
核心思路：复杂度为O(n)，空间复杂度为O(n)，需要考虑：原地操作的话修改本地字符会影响后续位置的字符
*/
#include <iostream>

using namespace std;

class Solution {
public:
    void ReplaceBlank(char *input_str, int length) {
        if (input_str == NULL || length <= 0) {
            return;
        }

        int blank_num = 0;
        int ori_offset = 0;
        while (ori_offset < length) {
            if (input_str[ori_offset] == ' ') {
                blank_num++;
            }
            ori_offset ++;
        }
        if (blank_num == 0) {
            return;
        }
        int new_length = length + 2 * blank_num;
        while (new_length > ori_offset && ori_offset >= 0) {
            if (input_str[ori_offset] == ' ') {
                input_str[new_length--] = '0';
                input_str[new_length--] = '2';
                input_str[new_length--] = '%';
            }
            else {
                input_str[new_length--] = input_str[ori_offset];
            }
            ori_offset--;
        }

    }
};


int main() {
    char input_str[] = "We are happy.";
    Solution s = Solution();
    s.ReplaceBlank(input_str, strlen(input_str));
    cout << "ReplaceBlank result: " << input_str << endl;
    return 0;
}