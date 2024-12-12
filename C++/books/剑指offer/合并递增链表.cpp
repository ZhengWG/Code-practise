/*
输入两个单调递增的链表，输出两个链表合成后的链表，
当然我们需要合成后的链表满足单调不减规则。
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
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2){
        // 类似归并排序
        ListNode* tmp = new ListNode(0);  // 归并后的链表虚拟节点
        ListNode* newList = tmp;  // 保存原始虚拟节点
        while (pHead1 && pHead2){
            if(pHead1->val < pHead2->val){
                tmp->next = pHead1;
                pHead1 = pHead1->next;
            }
            else{
                tmp->next = pHead2;
                pHead2 = pHead2->next;
            }
            tmp = tmp->next;
        }
        tmp->next = pHead1 ? pHead1 : pHead2;
        return newList->next;
    }
};
