/*  
输入两个树的节点，输出这两个节点的最低公共祖先
*/
#include <iostream>
#include <vector>
using namespace std;

struct TreeNode{
    int val;
    vector<TreeNode*> children; 
    TreeNode(int x) : val(x) {}
};

class Solution{
public:
    bool findPath(TreeNode* root, TreeNode* target, vector<TreeNode*> temp, vector<TreeNode*>& path){
        // 核心思路：递归找到target的路径
        if(root == NULL || target == NULL)
            return false;

        path.push_back(root);
        bool found = false;
        if (root == target) {
            return true;
        }
        for(auto child : root->children){
            found = findPath(child, target, temp, path);
            if (found)
                break;  
        }
        if (!found)
            path.pop_back();
        return found;
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // 核心思路：先找到p和q的路径，然后从root出发，找到第一个公共节点
        vector<TreeNode*> pathP;
        vector<TreeNode*> pathQ;
        vector<TreeNode*> temp;
        findPath(root, p, temp, pathP);
        findPath(root, q, temp, pathQ);

        // debug print pathP/pathQ
        for(auto node : pathP)
            cout << node->val << " ";
        cout << endl;
        for(auto node : pathQ)
            cout << node->val << " ";
        cout << endl;

        TreeNode* result = NULL;
        for(int i = 0; i < pathP.size() && i < pathQ.size(); i++){
            if(pathP[i] == pathQ[i])
                result = pathP[i];
            else
                break;
        }
        return result;
    }
};

int main(){
    Solution s;
    TreeNode* root = new TreeNode(1);
    root->children.push_back(new TreeNode(2));
    root->children.push_back(new TreeNode(3));
    root->children[0]->children.push_back(new TreeNode(4));
    root->children[0]->children.push_back(new TreeNode(5));
    root->children[1]->children.push_back(new TreeNode(6));
    root->children[1]->children.push_back(new TreeNode(7));
    root->children[0]->children[1]->children.push_back(new TreeNode(8));
    cout << s.lowestCommonAncestor(root, root->children[0]->children[0], root->children[0]->children[1]->children[0])->val << endl;
    return 0;
}   