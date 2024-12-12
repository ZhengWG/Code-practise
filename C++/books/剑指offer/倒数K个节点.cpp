/*
输入一个链表，输出该链表中倒数第k个结点。
*/
#include <iostream>
using namespace std;

struct ListNode {
    int val;
    struct ListNode* next;
    ListNode(int x): val(x), next(NULL) {}
};

class Solution{
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        // 快慢指针： 快指针领先慢指针k步，同时前进，直到快指针到达链表末尾
        // 边界条件
        if(pListHead == NULL || k == 0)
            return NULL;

        // TODO: 判断链表是否有环
        // 快指针向前
        ListNode* slow = pListHead;
        ListNode* fast = pListHead;
        for(int i=0; i<k; i++){
            if(fast->next != NULL)
                fast = fast->next;
            else
                return NULL;           
        }
        while(fast->next != NULL){
            fast = fast->next;
            slow = slow->next;
        }
        return slow;
    }
};

int main(){
    Solution s;
    ListNode* head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    ListNode* result1 = s.FindKthToTail(head, 1);
    cout << result1->val << endl;
    ListNode* result2 = s.FindKthToTail(head, 2);
    cout << result2->val << endl;
    return 0;
}