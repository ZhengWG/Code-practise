/*
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
(注意: 在返回值的list中，数组长度大的数组靠前)
*/
#include <iostream>
#include <vector>

using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x):val(x),left(NULL),right(NULL){}
};

class Solution{
public:
    void DFS(TreeNode* root, int expectNumber, vector<vector<int>>& out, vector<int>& tmp_paths, int& sum){
        // 边界条件
        if(!root)
            return;

        sum += root->val;
        tmp_paths.push_back(root->val);
        if(root->left == nullptr && root->right == nullptr){
            // 如果当前节点是叶子节点，检查路径和是否等于expectNumber
            if (sum == expectNumber){
                out.push_back(tmp_paths);
            }
            return;
        }
        
        // 递归遍历
        DFS(root->left, expectNumber, out, tmp_paths, sum);
        DFS(root->right, expectNumber, out, tmp_paths, sum);
        sum -= root->val;
        tmp_paths.pop_back();
        return;

    }
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber){
        // 核心思路：基于DFS的深度优先搜索
        vector<vector<int> > out;
        vector<int> tmp_paths;
        int sum = 0;
        DFS(root, expectNumber, out, tmp_paths, sum);
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
    vector<vector<int>> result = sol.FindPath(root, 7);
    for(auto path:result){
        for(int num:path)
            cout<<num<<" ";
        cout<<endl;
    }
    return 0;
}