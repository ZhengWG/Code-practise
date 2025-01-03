/*
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
*/
#include <iostream>
using namespace std;

struct TreeNode {
    int val;
    struct TreeNode* left;
    struct TreeNode* right;
    TreeNode(int x): val(x), left(NULL), right(NULL) {}
};

class Solution{
public:
    bool Judge(TreeNode* t1, TreeNode* t2){
        // 递归判断剩余子树是否相同

        // 边界条件
        if(!t1 && t2)
            return false;
        if(!t2)
            return true;
        if(t1->val != t2->val)
            return false;
        return Judge(t1->left, t2->left) && Judge(t1->right, t2->right);
    }
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2){
        // 核心思路是递归判断两个子树是否相同
        // 如果某节点相同，则递归判断两个子树是否相同
        // 如果不同则对pRoot1的左右子树继续递归判断

        bool flag = false;
        if (pRoot1 != NULL && pRoot2 != NULL){
            if (pRoot1->val == pRoot2->val)
                flag = Judge(pRoot1, pRoot2);
            if (!flag)
                flag = HasSubtree(pRoot1->left, pRoot2);
            if (!flag)
                flag = HasSubtree(pRoot1->right, pRoot2);
        }
        return flag;
    }
};