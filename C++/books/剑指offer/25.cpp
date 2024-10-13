/*
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
返回结果为复制后复杂链表的head。
（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
*/
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
	//思路：先double一下整个链表，再分割出来，复制double的时候，注意要分开
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(!pHead)
            return NULL;
        RandomListNode* curr = pHead;
        while(curr){
            RandomListNode* Head = new RandomListNode(curr->label);
            Head->next = curr->next;
            curr->next = Head;
            curr = Head->next;
        }
        curr = pHead;
        while(curr){
            RandomListNode* Head = curr->next;
            if(curr->random)
                Head->random = curr->random->next;
            curr = Head->next;
        }
        curr = pHead;
        RandomListNode* Head = new RandomListNode(0);
        RandomListNode* tmp = Head;
        while(curr){
            tmp->next = curr->next;
            tmp = tmp->next;
            curr->next = tmp->next;
            curr = curr->next;
        }
        tmp->next=NULL;
        return Head->next;
    }
};