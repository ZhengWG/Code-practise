//输出顺序不对
class Solution {
public:
    vector<vector<int>> out;
    vector<int> tmp;
    int max_j = 0;
    bool wordBreak1(string s, unordered_set<string> &dict) {
        bool dp[s.size()+1];
        for(int i=1;i<=s.size();i++)
            dp[i]=0;
        dp[0]=1;
        for(int i=1;i<=s.size();i++){
            for(int j=i;j<=s.size();j++){
                string tmp = s.substr(i-1,j-i+1);
                auto p = dict.end();
                if(dict.find(tmp)!=p && dp[i-1]){
                    dp[j]=1;
                }
            }
        }
        return dp[s.size()];
    }
    void DFS(string s, bool** dp,int a,int b, int n){
        if(b==n){
            out.push_back(tmp);
            return;
        }
        for(int i=b;i<=max_j;i++){
            if(dp[a][i]){
                tmp.push_back(i);
                DFS(s,dp,i+1,i+1,n);
                tmp.pop_back();
            }
        }
        return;
    }
    vector<string> wordBreak(string s, unordered_set<string> &dict) {
        vector<string> s_s;
        if(!wordBreak1(s,dict))
            return s_s;
        bool** dp =new bool* [s.size()+1];
        for(int i=0;i<=s.size();i++)
            dp[i] = new bool [s.size()+1];
        for(int i=0;i<=s.size();i++){
            for(int j=0;j<s.size();j++)
                dp[i][j]=0;
        }
        for(int i=0;i<s.size();i++){
            for(int j=i;j<s.size();j++){
                string tmp = s.substr(i,j-i+1);
                auto it = dict.end();
                if(dict.find(tmp)!=it){
                    dp[i][j]=1;
                    max_j = j;
                }
            }
        }
        //对结果进行深度遍历
        DFS(s, dp,0,0,s.size());
        for(int i=0;i<=s.size();i++)
            delete[] dp[i];
        delete[] dp;
        for(int i=out.size()-1;i>=0;i--){
            string tmp = "";
            int pre = 0;
            for(int j=0;j<out[i].size();j++){
                tmp.append(s.substr(pre,out[i][j]-pre+1));
                if(j!=out[i].size()-1)
                    tmp.append(" ");
                pre = out[i][j]+1;
            }
            s_s.push_back(tmp);
        }
        return s_s;
    }
};