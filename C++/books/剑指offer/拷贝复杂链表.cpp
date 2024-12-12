/*
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
返回结果为复制后复杂链表的head。
（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
*/
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

struct RandomListNode{
    int label;
    RandomListNode* next;
    RandomListNode* random;
    RandomListNode(int x):label(x),next(nullptr),random(nullptr){}
};

class Solution{
public:
    RandomListNode* Clone(RandomListNode* pHead){
        // 关键在于random指针的拷贝，思路： 
        // 1. 先double一下整个链表，再分割出来
        // 2. 复制double的时候，注意要分开
        // 3. 分割的时候，注意要分开

        // 边界条件
        if(!pHead)
            return NULL;

        // 插入拷贝指针
        RandomListNode* curr = pHead;
        while(curr){
            RandomListNode* Head = new RandomListNode(curr->label);
            // 插入拷贝节点到中间
            Head->next = curr->next;
            curr->next = Head;
            curr = Head->next;
        }

        // 指向random指针
        curr = pHead;
        while(curr){
            RandomListNode* Head = curr->next;
            if(curr->random)
                Head->random = curr->random->next;
            curr = Head->next;
        }

        // 分割: 需要新建HEAD指针指向拷贝后的指针
        curr = pHead;
        RandomListNode* Head = new RandomListNode(0);
        RandomListNode* tmp = Head;
        while(curr){
            // tmp/curr后移2步
            RandomListNode* Head = curr->next;
            tmp->next = Head;
            tmp = Head;
            curr = Head->next;
        }
        return Head->next;
    }
};
