/*
将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0
    */
#include <iostream>
#include <string>
using namespace std;

class Solution{
public:
    bool InvalidFlag = false;
    int StrToInt(string str) {
        if(str.empty()){
            InvalidFlag = true;
            return 0;
        }

        int sign = 1;
        int start = 1;
        if(str[0] == '-')
            sign = -1;
        else if(str[0] == '+')
            sign = 1;
        else
            start = 0;
        long long result = 0;
        for(int i = start; i < str.size(); i++){
            if(str[i] < '0' || str[i] > '9'){
                InvalidFlag = true;
                return 0;
            }
            result = result * 10 + (str[i] - '0');
            if(result > INT_MAX){
                InvalidFlag = true;
                return 0;
            }
        }
        return result * sign;
    }
};

int main(){
    Solution s;
    cout << s.StrToInt("123") << endl;
    cout << s.StrToInt("-123") << endl;
    cout << s.StrToInt("123a") << endl;
    return 0;
}