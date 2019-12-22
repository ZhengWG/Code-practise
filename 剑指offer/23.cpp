/*
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
*/
class Solution {
public:
    bool Judge(vector<int> num,int l, int r){
        if(l>=r)
            return true;
        int mid = r-1;
        while(mid>=l&&num[mid]>num[r])
            mid--;
        for(int j = mid;j>=l;j--)
            if(num[j]>num[r])
                return false;
        return Judge(num,l,mid) && Judge(num,mid+1,r-1);
    }
    bool VerifySquenceOfBST(vector<int> sequence) {
        if(!sequence.size())
            return false;
        return Judge(sequence,0,sequence.size()-1);
    }
};