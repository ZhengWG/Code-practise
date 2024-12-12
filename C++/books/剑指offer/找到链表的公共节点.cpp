/*
输入两个链表，找出它们的第一个公共结点。
*/
#include <iostream>
using namespace std;

struct ListNode{
    int val;
    ListNode* next;
    ListNode(int x): val(x), next(NULL){}
};

class Solution{
public:
    int GetLength(ListNode* pHead){
        int len = 0;
        ListNode* p = pHead;
        while(p){
            p = p->next;
            len++;
        }
        return len;
    }

    ListNode* FindFirstCommonNode(ListNode* pHead1, ListNode* pHead2) {
        // 核心思路：先找到两个链表的长度，然后让长的链表先走长度差，然后两个链表一起走，直到找到公共节点
        int len1 = GetLength(pHead1);
        int len2 = GetLength(pHead2);
        int diff = len1 - len2;
        ListNode* p1 = pHead1;
        ListNode* p2 = pHead2;
        if(len1 < len2){
            p1 = pHead2;
            p2 = pHead1;
            diff = len2 - len1;
        }
        while(diff > 0){
            p1 = p1->next;
            diff--;
        }
        while(p1 && p2 && p1 != p2){
            p1 = p1->next;
            p2 = p2->next;
        }
        return p1;
    }
};

int main(){
    Solution sol;
    ListNode* pHead1 = new ListNode(1);
    pHead1->next = new ListNode(2);
    ListNode* pHead2 = new ListNode(3);
    ListNode* common_node = new ListNode(4);
    common_node->next = new ListNode(5);
    pHead1->next->next = common_node;
    pHead2->next = common_node;
    cout << sol.FindFirstCommonNode(pHead1, pHead2)->val << endl;
    return 0;
}