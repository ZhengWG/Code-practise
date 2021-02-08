/*
输入一棵二叉树，判断该二叉树是否是平衡二叉树。
*/
class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        int depth;
        return getDepth(pRoot, &depth);
    }
	//关键在于递归求深度
    bool getDepth(TreeNode *root, int *depth){
		if (!root){
			*depth = 0;
			return 1;
		}
		int l_depth;
		int r_depth;
		bool l_ba = getDepth(root->left, &l_depth);
		bool r_ba = getDepth(root->right, &r_depth);
		if (l_ba && r_ba){
			int l_diff = l_depth - r_depth;
			if (l_diff <= 1 && l_diff>=-1){
				*depth = (l_depth > r_depth ? l_depth : r_depth) + 1;
				return true;
			}
		}
		return false;
	}
};