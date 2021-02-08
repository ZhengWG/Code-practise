/*
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
*/

/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> ori;
        while(head != NULL){
            ori.push_back(head->val);
            head = head->next;
        }
        vector<int> out;
        for(int i=ori.size()-1;i>=0;i--){
            out.push_back(ori[i]);
        }
        return out;
    }
};