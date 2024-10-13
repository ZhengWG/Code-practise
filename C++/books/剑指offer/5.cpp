/*
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
*/

class Solution
{
public:
    void push(int node) {
		//无论压入还是弹出都需要保证另一个堆栈为空
        if(stack2.empty()){
            stack1.push(node);
        }
        else{
            while(!stack2.empty()){
                stack1.push(stack2.top());
                stack2.pop();
            }
            stack1.push(node);
        }
    }
    
    int pop() {
        while(!stack1.empty()){
            stack2.push(stack1.top());
            stack1.pop();
        }
        int node = stack2.top();
        stack2.pop();
        return node;
    }

private:
    stack<int> stack1;//用作压入使用
    stack<int> stack2;//用作弹出使用
};