/*
输入一棵二叉树，求该树的深度。
从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
*/
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
	//深度遍历
    int final_max;
    int tmp_max;
    int TreeDepth(TreeNode* pRoot)
    {
        if(!pRoot)
            return 0;
        final_max = 1;
        tmp_max= 1;
        getDepth(pRoot);
        return final_max;
    }
    void getDepth(TreeNode *root){
        if(!root->left && !root->right){
            if(tmp_max>final_max)
                final_max=tmp_max;
            return;
        }
        if(root->left){
            tmp_max++;
            getDepth(root->left);
            tmp_max--;
        }
        if(root->right){
            tmp_max++;
            getDepth(root->right);
            tmp_max--;
        }
        return;
    }
};