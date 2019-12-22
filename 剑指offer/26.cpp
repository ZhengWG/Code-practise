/*
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
要求不能创建任何新的结点，只能调整树中结点指针的指向。
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
	TreeNode* lastnode = NULL;
	TreeNode* Convert(TreeNode* pRootOfTree)
	{
		NodeConvert(pRootOfTree);
		while (lastnode && lastnode->left)
			lastnode = lastnode->left;
		return lastnode;
	}
	//中序遍历，lastnode记录的是前一个结点
	void NodeConvert(TreeNode* root){
		if (!root)
			return;
		if (root->left != NULL)
			NodeConvert(root->left);
		root->left = lastnode;
		if (lastnode != NULL)
			lastnode->right = root;
		lastnode = root;
		if (root->right != NULL)
			NodeConvert(root->right);
		return;

	}
};