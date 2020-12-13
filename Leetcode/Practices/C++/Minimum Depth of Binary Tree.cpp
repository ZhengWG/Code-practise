class Solution {
public:
    int run(TreeNode *root) {
        //BFS
        if(root==NULL)
            return 0;
        queue<TreeNode*> q;
        queue<int> layer;
        q.push(root);
        layer.push(1);
        while(!q.empty()){
            TreeNode* tmp = q.front();
            q.pop();
            int la = layer.front()+1;
            if(tmp->left==NULL && tmp->right==NULL)
                break;
            else{
                if(tmp->left!=NULL){
                    q.push(tmp->left);
                    layer.push(la);
                }
                if(tmp->right!=NULL){
                    q.push(tmp->right);
                    layer.push(la);
                }
            }
            layer.pop();
        }
        return layer.front();
    }
};