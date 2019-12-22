/*
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
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
	//常规层遍历
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        vector<int> out;
        if(!root)
            return out;
        queue<TreeNode *> tmp;
        tmp.push(root);
        while(!tmp.empty()){
            TreeNode* node = tmp.front();
            out.push_back(node->val);
            tmp.pop();
            if(node->left)
                tmp.push(node->left);
            if(node->right)
                tmp.push(node->right);
        }
        return out;
    }
};