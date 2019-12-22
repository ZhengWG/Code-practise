/*
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
*/

/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        if(!pre.size())
            return NULL;
        TreeNode* root = new TreeNode(pre[0]);
        int k;
		//关键在于找到中序遍历和前序遍历的同一点，根据其来重建左右子树序列
        for(k=0;k<vin.size();k++){
            if(vin[k]==pre[0])
                break;
        }
        vector<int> Lpre_new, Lvin_new, Rpre_new, Rvin_new;
        for(int i=1;i<=k;i++){
            Lpre_new.push_back(pre[i]);
            Lvin_new.push_back(vin[i-1]);
        }
        for(int i=k+1;i<pre.size();i++){
            Rpre_new.push_back(pre[i]);
            Rvin_new.push_back(vin[i]);
        }
        root->left = reConstructBinaryTree(Lpre_new, Lvin_new);
        root->right = reConstructBinaryTree(Rpre_new, Rvin_new);
        return root;
    }
};