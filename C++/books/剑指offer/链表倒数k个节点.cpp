/*
输入一个链表，输出该链表中倒数第k个结点。
*/
#include <iostream>


struct ListNode {
	int val;
	struct ListNode* next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};


class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        if (pListHead == NULL || k == 0)
            return NULL;

        ListNode* pa = pListHead;
        ListNode* pb = pListHead;
        for (int i=0; i<k-1; i++) {
            if (pa->next != NULL)
                pa = pa->next;
            else
                return NULL;
        }
        while (pa->next != NULL) {
            pa = pa->next;
            pb = pb->next;
        }
        return pb;
    }
}