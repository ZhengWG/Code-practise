/*
输入两个单调递增的链表，输出两个链表合成后的链表，
当然我们需要合成后的链表满足单调不减规则。
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
	//其实蕾丝衫与归并排序，注意首指针的细节处理
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        ListNode* tmp = new ListNode(0);
        ListNode* newList = tmp;
        while(pHead1 && pHead2){
            if(pHead1->val < pHead2->val){
                tmp->next = new ListNode(pHead1->val);
                tmp = tmp->next;
                pHead1 = pHead1->next;
            }
            else{
                tmp->next = new ListNode(pHead2->val);
                tmp = tmp->next;
                pHead2 = pHead2->next;
            }
        }
        while(pHead1){
            tmp->next = new ListNode(pHead1->val);
            tmp = tmp->next;
            pHead1 = pHead1->next;
        }
        while(pHead2){
            tmp->next = new ListNode(pHead2->val);
            tmp = tmp->next;
            pHead2 = pHead2->next;
        }
        ListNode* out  = newList->next;
        return out;
    }
};
