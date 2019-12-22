/*
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
*/
class Solution {
public:
	//自加比较很灵性
    static bool cmp(int in1, int in2){
        string a = "";
        string b = "";
        a += to_string(in1);
        a += to_string(in2);
        b += to_string(in2);
        b += to_string(in1);
        return a<b;
    }
    string PrintMinNumber(vector<int> numbers) {
        sort(numbers.begin(),numbers.end(),cmp);
        string out = "";
        for(int i=0;i<numbers.size();i++){
            out += to_string(numbers[i]);
        }
        return out;
    }
};