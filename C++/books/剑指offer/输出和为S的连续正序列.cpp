/*
小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。
但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。
没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!
输出描述:
输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序。
*/
#include <iostream>
using namespace std;

class Solution{
public:
    vector<vector<int>> FindContinuousSequence(int sum){
        // 核心思路：滑动窗口，窗口大小从2开始
        vector<vector<int>> out;
        int phigh = 2, plow = 1;
        while(phigh > plow and phigh < sum){
            int re = (phigh + plow)*(phigh - plow + 1)/2;
            if(re == sum){
                vector<int> o;
                for(int i = plow; i<=phigh; i++){
                    o.push_back(i);
                }
                out.push_back(o);
                plow++;
            }
            else if(re < sum)
                phigh++;
            else
                plow++;
        }
        return out;
    }
};

int main(){
    Solution s;
    vector<vector<int>> out = s.FindContinuousSequence(100);
    for(auto &o : out){
        for(auto &i : o) cout << i << " ";
        cout << endl;
    }
    return 0;
}