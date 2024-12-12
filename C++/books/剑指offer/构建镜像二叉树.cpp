/*
操作给定的二叉树，将其变换为源二叉树的镜像。
输入描述:
二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
*/
#include <iostream>
using namespace std;

struct TreeNode{
    int val;
    struct TreeNode* left;
    struct TreeNode* right;
    TreeNode(int x): val(x), left(NULL), right(NULL) {}
};

class Solution{
public:
    void Mirror_Recursive(TreeNode* pRoot){
        // 核心思路是：从根节点开始，进行递归交换，从上往下交换

        // 边界条件
        if(pRoot == NULL)
            return;

        // 交换当前节点左右子树
        TreeNode* tmp = pRoot->left;
        pRoot->left = pRoot->right;
        pRoot->right = tmp;

        // 递归左右子树交换
        Mirror_Recursive(pRoot->left);
        Mirror_Recursive(pRoot->right);
    }
    
    void Mirror_Stack(TreeNode* pRoot) {
        // 边界条件
        if (pRoot == NULL)
            return;
            
        // 使用栈来存储需要处理的节点
        stack<TreeNode*> st;
        st.push(pRoot);
        
        // 当栈不为空时，继续处理
        while (!st.empty()) {
            // 取出当前需要处理的节点
            TreeNode* current = st.top();
            st.pop();
            
            // 交换当前节点的左右子树
            TreeNode* tmp = current->left;
            current->left = current->right;
            current->right = tmp;
            
            // 将左右子节点压入栈中（如果存在的话）
            if (current->left)
                st.push(current->left);
            if (current->right)
                st.push(current->right);
        }
    }
};