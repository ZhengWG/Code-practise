/*
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
*/
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    // 递归判断序列在区间[l,r]内是否是BST的后序遍历
    // Recursively verify if sequence[l...r] is a valid BST post-order traversal
    bool Judge(vector<int> num, int l, int r) {
        // 如果左边界大于等于右边界，说明区间为空或只有一个节点，返回true
        // Base case: if interval is empty or contains one element
        if(l >= r)
            return true;
            
        // 在后序遍历中，最后一个元素是根节点
        // mid用于找到左右子树的分界点
        int mid = r - 1;
        
        // 从右向左找到第一个小于根节点的位置
        // 这个位置左边是左子树（都小于root），右边是右子树（都大于root）
        while(mid >= l && num[mid] > num[r])
            mid--;
            
        // 检查左子树中的所有节点是否都小于根节点
        // Verify all nodes in left subtree are smaller than root
        for(int j = mid; j >= l; j--)
            if(num[j] > num[r])
                return false;
                
        // 递归判断左子树和右子树是否都是合法的后序遍历
        // Recursively verify both left and right subtrees
        return Judge(num, l, mid) &&    // 验证左子树 verify left subtree
               Judge(num, mid+1, r-1);  // 验证右子树 verify right subtree
    }
    
    // 验证序列是否是BST的后序遍历的入口函数
    // Entry function to verify if sequence is a valid BST post-order traversal
    bool VerifySquenceOfBST(vector<int> sequence) {
        // 空序列返回false
        // Empty sequence is not valid
        if(!sequence.size())
            return false;
            
        // 调用递归函数进行判断
        // Call recursive helper function
        return Judge(sequence, 0, sequence.size()-1);
    }
};