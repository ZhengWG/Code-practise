/*
在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,
并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
*/
#include <iostream>
#include <string>
#include <map>
using namespace std;

class Solution{
public:
    char FirstNotRepeatingChar(string str) {
        map<char, int> char_count;
        for(char c : str){
            char_count[c]++;
        }
        for(auto it = char_count.begin(); it != char_count.end(); it++){
            if(it->second == 1) return it->first;
        }
        return -1;
    }   
};

int main(){
    Solution sol;
    string str = "abacb";
    char result = sol.FirstNotRepeatingChar(str);
    cout << result << endl;
    return 0;
}