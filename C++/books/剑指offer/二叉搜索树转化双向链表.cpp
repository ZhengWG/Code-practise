/*
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
要求不能创建任何新的结点，只能调整树中结点指针的指向。
*/
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x):val(x),left(nullptr),right(nullptr){}
};

class Solution{
public:
    TreeNode* lastnode = nullptr; // 记录上一个节点
    TreeNode* Convert(TreeNode* pRootOfTree){
        NodeConvert(pRootOfTree);
        while(lastnode && lastnode->left)
            lastnode = lastnode->left;
        return lastnode;
    }

    void NodeConvert(TreeNode* pRootOfTree){
        // 核心思路：
        // 二叉搜索树的中序遍历结果有序，可基于中序遍历递归，递归中构建双向链表
        // 1. 递归左子树
        // 2. 递归右子树
        // 3. 递归中构建双向链表，需要记录前一个节点

        // 边界条件
        if(!pRootOfTree)
            return;

        // 递归左子树
        NodeConvert(pRootOfTree->left);

        // 递归中构建双向链表
        if(lastnode){
            lastnode->right = pRootOfTree;
            pRootOfTree->left = lastnode;
        }
        lastnode = pRootOfTree;

        // 递归右子树
        NodeConvert(pRootOfTree->right);
    }
};

int main(){
    Solution sol;
    TreeNode* root = new TreeNode(10);
    root->left = new TreeNode(6);
    root->left->left = new TreeNode(4);
    root->left->left->left = new TreeNode(3);
    root->right = new TreeNode(14);
    TreeNode* result = sol.Convert(root);
    return 0;
}