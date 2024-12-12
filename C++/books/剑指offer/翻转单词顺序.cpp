/*
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。
为简单起见，标点符号和普通字母一样处理。
例如输入字符串"I am a student."，则输出"student. a am I"。
*/
#include <iostream>
#include <string>
using namespace std;

class Solution{
public:
    string ReverseSentence(string str) {
        // 核心思路：
        // 1. 先翻转整个字符串
        // 2. 再翻转每个单词
        // 3. 最后返回结果
        reverse(str.begin(), str.end());

        // 翻转每个单词
        int start = 0, end = 0;
        while(end < str.size()){
            if(str[end] == ' '){
                reverse(str.begin() + start, str.begin() + end);
                start = end + 1;
            }
            end++;
        }
        return str;
    }

    string ReverseLeftWords(string str, int n) {
        // 核心思路：
        // 1. 先翻转前n个字符
        // 2. 再翻转后面的字符
        // 3. 最后返回结果
        reverse(str.begin(), str.begin() + n);
        reverse(str.begin() + n, str.end());
        reverse(str.begin(), str.end());
        return str;
    }
};

int main(){
    Solution s;
    string str = "I am a student.";
    cout << s.ReverseSentence(str) << endl;
    cout << s.ReverseLeftWords(str, 2) << endl;
    return 0;
}