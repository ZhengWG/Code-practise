/*
输入一个字符串,按字典序打印出该字符串中字符的所有排列。
例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
输入描述:
输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
*/
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    void Get(int start, string str, vector<string> &out) {
        // 递归终止条件：当start到达字符串末尾时
        if (start == str.length() - 1) {
            out.push_back(str);
            return;
        }
        
        // 记录当前位置是否已经交换过相同的字符
        vector<bool> used(256, false);
        
        // 依次与后面的字符交换位置
        for (int i = start; i < str.length(); i++) {
            // 如果当前字符已经在该位置使用过，跳过
            if (used[str[i]]) continue;
            used[str[i]] = true;
            
            // 交换字符
            swap(str[start], str[i]);
            // 递归处理剩余子串
            Get(start + 1, str, out);
            // 恢复交换
            swap(str[start], str[i]);
        }
    }
};