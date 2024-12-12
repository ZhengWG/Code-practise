/*
输入一棵二叉树，求该树的深度。
从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
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
    int TreeDepth_DFS(TreeNode* pRoot){
        // 核心思路：DFS获取深度
        if(!pRoot) return 0;
        return max(TreeDepth_DFS(pRoot->left), TreeDepth_DFS(pRoot->right)) + 1;
    }

    int TreeDepth_BFS(TreeNode* pRoot){
        // 核心思路：BFS获取深度
        if(!pRoot) return 0;
        queue<TreeNode*> q;
        q.push(pRoot);
        int depth = 0;
        while(!q.empty()){
            int size = q.size();
            for(int i = 0; i < size; i++){
                TreeNode* node = q.front();
                q.pop();
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
            }
            depth++;
        }
        return depth;
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
    root->left->left->left = new TreeNode(8);
    cout << "DFS: " << s.TreeDepth_DFS(root) << endl;
    cout << "BFS: " << s.TreeDepth_BFS(root) << endl;
    return 0;
}