/*
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
*/
#include <iostream>
#include <vector>

using namespace std;


struct ListNode {
    int m_nValue;
    ListNode* m_pNext;
};


class Solution {
public:
    void PrintListReversing_Iter(ListNode* pHead) {
        stack<ListNode*> nodes;
        ListNode* pNode = pHead;
        while (pNode != NULL) {
            nodes.push(pNode);
            pNode = pNode->m_pNext;
        }
        while (!nodes.empty()) {
            pNode = nodes.top();
            printf("%d\t", pNode->m_nValue);
            nodes.pop();
        }
    }
};


int main() {
    int node_num = 5;
    vector<ListNode*> nodes;
    for (int i=0; i<node_num; i++){
        ListNode* node = new ListNode();
        node->m_nValue = i;
        node->m_pNext = NULL;
        nodes.push_back(node);
    }
    for (int i=0; i<node_num-1; i++){
        ListNode* node = nodes[i];
        ListNode* next_node = nodes[i+1];
        node->m_pNext = next_node;
    }
    Solution s = Solution();
    s.PrintListReversing_Iter(nodes[0]);
}