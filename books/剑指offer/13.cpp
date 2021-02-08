/*
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
并保证奇数和奇数，偶数和偶数之间的相对位置不变。
*/
class Solution {
public:
	//没啥特别的，关键在于用与运算进行奇偶性判断
    void reOrderArray(vector<int> &array) {
        vector<int> out;
        for(int i=0;i<array.size();i++){
            if(array[i]&0x1==1)
                out.push_back(array[i]);
        }
        for(int i=0;i<array.size();i++){
            if((array[i]&0x1)!=1)
                out.push_back(array[i]);
        }
        array  = out;
    }
};