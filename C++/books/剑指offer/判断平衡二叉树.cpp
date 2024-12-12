/*
输入一棵二叉树，判断该二叉树是否是平衡二叉树。
*/
#include <iostream>
using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x): val(x), left(NULL), right(NULL){}
};

class Solution{
public:
    int TreeDepth(TreeNode* pRoot){
        // 核心思路：递归获取深度
        if(!pRoot) return 0;
        return max(TreeDepth(pRoot->left), TreeDepth(pRoot->right)) + 1;
    }
    bool IsBalanced_Solution(TreeNode* pRoot) {
        // 核心思路：递归判断左右子树是否平衡，然后判断左右子树的深度差是否小于等于1
        if(!pRoot) return true;
        int left_depth = TreeDepth(pRoot->left);
        int right_depth = TreeDepth(pRoot->right);
        cout << "left_depth: " << left_depth << ", right_depth: " << right_depth << endl;
        if(abs(left_depth - right_depth) > 1) return false;
        return IsBalanced_Solution(pRoot->left) && IsBalanced_Solution(pRoot->right);
    }
};

int main(){
    Solution s;
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(7);
    cout << s.IsBalanced_Solution(root) << endl;
    return 0;
}