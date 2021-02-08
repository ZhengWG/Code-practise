/*
输入一个字符串,按字典序打印出该字符串中字符的所有排列。
例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
输入描述:
输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
*/
class Solution {
public:
    void Get(int start, string str, vector<string> &out){
        if(start==str.size()-1)
            out.push_back(str);
        sort(str.begin()+start,str.end());
		//list用来保证不会出现重复数字的交换
        unordered_set <char> list;
        for(int i=start;i<str.size();i++){
            if(list.find(str[i])==list.end()){
                list.insert(str[i]);
                swap(str[i],str[start]);
                Get(start+1,str,out);
                swap(str[i],str[start]);
            }
        }
    }
    vector<string> Permutation(string str) {
        vector<string> out;
        Get(0,str,out);
        return out;
    }
};