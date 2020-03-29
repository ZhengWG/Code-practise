"""
输入：两个有序数组
输出：两个有序数组合并后的中位数
输入示例:[1,2],[3,4]
输出示例:2.5
输入示例:[1,2],[4]
输出示例:2
要求时间复杂度为log(m+n)，其中m,n为两个数组的长度
"""

nums1 = [1,5]
nums2 = [2, 3, 4, 6]

def findMedianSortedArrays(nums1, nums2):
    m, n = len(nums1), len(nums2)
    if m > n:
        nums1, nums2, m, n = nums2, nums1, n, m
    imin, imax, half_len = 0, m, (m+n+1)/2
    while imin <= imax:
        #i为m数组的划分idx，j为n数组的划分idx
        i = int((imin + imax) / 2)
        j = int(half_len - i)
        #二分,将两个数组分为两个部分
        if i < m and  nums2[j-1] > nums1[i]:
            imin = i + 1
        elif i > 0 and  nums1[i-1] > nums2[j]:
            imax = i - 1
        else:
            #四种边界情况
            if i == 0:
                #说明nums1都在右边
                max_of_left = nums1[j-1]
            elif j == 0:
                #说明nums2都在右边
                max_of_left = nums1[i-1]
            else:
                max_of_left = max(nums1[i-1], nums2[j-1])
            if (m + n) % 2 == 1:
                return max_of_left
            if i == m:
                #说明nums1都在左边
                min_of_right = nums2[j]
            elif j == n:
                #说明nums2都在左边
                min_of_right = nums2[i]
            else:
                min_of_right = min(nums1[i], nums2[j])
            return (max_of_left + min_of_right) / 2

print(findMedianSortedArrays(nums1, nums2))
