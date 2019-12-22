/*
输入两个链表，找出它们的第一个公共结点。
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
    //快慢指针，需要先得到链表长度差
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        int l1 = Getlength(pHead1);
        int l2 = Getlength(pHead2);
        int l_d;
        ListNode* pl;
        ListNode* ps;
        if(l1>l2){
            l_d = l1-l2;
            pl = pHead1;
            ps = pHead2;
        }
        else{
            l_d = l2-l1;
            pl = pHead2;
            ps = pHead1;
        }
        while(l_d){
            pl = pl->next;
            l_d--;
        }
        while(pl && ps && ps!=pl){
            pl = pl->next;
            ps = ps->next;
        }
        return pl;
    }
    int Getlength(ListNode* list){
        int le = 0;
        ListNode* node = list;
        while(node){
            node = node->next;
            le++;
        }
        return le;
    }
};