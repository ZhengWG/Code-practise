class Solution {
public:
    int majorityElement(vector<int>& nums) {
        if(nums.empty()) return 0;
        
        int start = 0;
        int end = nums.size() - 1;
        int middle = nums.size() >> 1;
        
        int index = partition(nums, start, end);
        while(index != middle) {
            if(index > middle) {
                end = index - 1;
            } else {
                start = index + 1;
            }
            index = partition(nums, start, end);
        }
        
        // Verify if the number appears more than n/2 times
        int count = 0;
        for(int num : nums) {
            if(num == nums[middle]) count++;
        }
        
        return count > nums.size()/2 ? nums[middle] : 0;
    }
    
private:
    int partition(vector<int>& nums, int start, int end) {
        int pivot = nums[start];
        
        while(start < end) {
            while(start < end && nums[end] >= pivot) end--;
            nums[start] = nums[end];
            
            while(start < end && nums[start] <= pivot) start++;
            nums[end] = nums[start];
        }
        
        nums[start] = pivot;
        return start;
    }
};
