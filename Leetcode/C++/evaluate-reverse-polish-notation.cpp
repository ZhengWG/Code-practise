
class Solution {
public:
    int evalRPN(vector<string> &tokens) {
        if(!tokens.size())
            return 0;
        stack<int> s;
        for(int i=0;i<tokens.size();i++){
            if(tokens[i]=="+" ||tokens[i]=="-" ||tokens[i]=="/" ||tokens[i]=="*"){
                int a = s.top();
                s.pop();
                int b = s.top();
                s.pop();
                if(tokens[i]=="+")
                    s.push(b+a);
                if(tokens[i]=="-")
                    s.push(b-a);
                if(tokens[i]=="*")
                    s.push(b*a);
                if(tokens[i]=="/")
                    s.push(b/a);
            }
            else
                s.push(atoi(tokens[i].c_str()));
       }
       return s.top();
    }
};