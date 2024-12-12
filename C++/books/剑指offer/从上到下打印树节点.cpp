/*
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
*/

#include<iostream>
#include<queue>
#include<vector>

using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x):val(x),left(NULL),right(NULL){}
};

class Solution{
public:
    vector<int> PrintFromTopToBottom(TreeNode* root){
        // 通过队列进行层序遍历
        if(!root)
            return {};
        vector<int> out;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            TreeNode* node = q.front();
            q.pop();
            out.push_back(node->val);  // 将当前节点值加入结果
            if(node->left)
                q.push(node->left);
            if(node->right)
                q.push(node->right);
        }
        return out;
    }
};

int main(){
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    Solution sol;
    vector<int> result = sol.PrintFromTopToBottom(root);
    for(int num:result)
        cout<<num<<" ";
    cout<<endl;
    return 0;
}