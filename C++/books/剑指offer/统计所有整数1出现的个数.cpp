/*
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。
ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
*/
#include <iostream>
using namespace std;

class Solution{
public:
    int NumberOf1Between1AndN_Solution(int n) {
        // 核心思路：从低位到高位，逐位统计1的个数
        if (n <= 0) return 0;
        
        int count = 0;
        int base = 1;
        int current, before, after;
        
        while (n >= base) {
            current = (n / base) % 10;   // 当前位数字
            before = n / (base * 10);    // 高位数字
            after = n % base;            // 低位数字
            
            // 根据当前位的数字分三种情况讨论
            if (current == 0) {
                count += before * base;
            } else if (current == 1) {
                count += before * base + after + 1;
            } else {
                count += (before + 1) * base;
            }
            
            base *= 10;
        }
        
        return count;
    }
};