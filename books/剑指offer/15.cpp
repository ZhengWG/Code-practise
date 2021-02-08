/*
输入一个链表，反转链表后，输出新链表的表头。
*/
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
	//将相邻链表元素调换即可，主要需要保留原有的表头
    ListNode* ReverseList(ListNode* pHead) {
        if(!pHead)
            return NULL;
        ListNode* before = pHead;
        ListNode* after = pHead->next;
        while(after){
            ListNode* tmp = after->next;
            after->next = before;
            before = after;
            after = tmp;
        }
        pHead->next = NULL;
        return before;
    }
};