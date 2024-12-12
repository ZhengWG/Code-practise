/*
整体思路：
这道题的难点在于如何处理复杂链表中的随机指针（m_pSibling）。如果用传统的方式直接复制，
很难在O(n)时间内找到随机指针指向的节点的复制节点。因此采用以下三步法：

1. 复制节点：在原链表的每个节点后面插入复制的节点
   原始：A -> B -> C
   复制后：A -> A' -> B -> B' -> C -> C'

2. 设置随机指针：利用相邻关系，可以通过原节点的random指针找到对应复制节点的random指针
   如果原始节点N的random指向S，则N'的random应指向S'
   而S'就是S的next指针指向的节点

3. 拆分链表：将原始链表和复制链表分离
   奇数位置的节点属于原始链表，偶数位置的节点属于复制链表
*/

ComplexListNode* Clone(ComplexListNode* pHead) {
    if (!pHead) return nullptr;  // 处理空链表的情况

    // Step 1: 复制节点并插入到原节点后面
    ComplexListNode* current = pHead;
    while (current) {
        // 创建新节点并复制数据
        ComplexListNode* cloned = new ComplexListNode();
        cloned->m_value = current->m_value;
        
        // 插入新节点到原节点后面
        cloned->m_pNext = current->m_pNext;
        current->m_pNext = cloned;
        
        // 移动到下一个原始节点
        current = cloned->m_pNext;
    }

    // Step 2: 设置复制节点的随机指针
    current = pHead;
    while (current) {
        // 如果原始节点有随机指针，则设置复制节点的随机指针
        if (current->m_pSibling) {
            // current->m_pSibling->m_pNext 就是随机指针指向节点的复制节点
            current->m_pNext->m_pSibling = current->m_pSibling->m_pNext;
        }
        // 移动到下一个原始节点
        current = current->m_pNext->m_pNext;
    }

    // Step 3: 分离原始链表和复制链表
    current = pHead;
    ComplexListNode* clonedHead = pHead->m_pNext;  // 保存复制链表的头节点
    while (current) {
        // 保存当前节点的复制节点
        ComplexListNode* cloned = current->m_pNext;
        
        // 恢复原始链表的next指针
        current->m_pNext = cloned->m_pNext;
        
        // 设置复制链表的next指针（如果不是最后一个节点）
        if (cloned->m_pNext) {
            cloned->m_pNext = cloned->m_pNext->m_pNext;
        }
        
        // 移动到下一个原始节点
        current = current->m_pNext;
    }

    return clonedHead;  // 返回复制链表的头节点
}
