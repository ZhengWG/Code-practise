/*
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
(注意: 在返回值的list中，数组长度大的数组靠前)
*/
#include <iostream>
#include <vector>

using namespace std;


struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};


class Solution {
public:
    void DFS(TreeNode* root, int num, vector<vector<int> > &out, vector<int> tmp, int sum) {
        if (!root)
            return;
        sum += root->val;
        tmp.push_back(root->val);
        if (!root->left && !root->right)
            if (sum == num)
                out.push_back(tmp);
        DFS(root->left, num, out, tmp, sum)
        DFS(root->right, num, out, tmp, sum)
        sum -= root->val;
        tmp.pop_back();
        return;
    }

    vector<vector<int> > FindPath(TreeNode* root, int exepectNumber) {
        vector<vector<int> > out;
        vector<int> tmp;
        int sum = 0;
        DFS(root, exepectNumber, out, tmp, sum);
        return out;
    }
};