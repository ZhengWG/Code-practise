/*
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 
输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
*/
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int left = 0,right=rotateArray.size()-1;
        int mid = left;
		//使用二分法进行判断，关键在于三个数相等的时候，需要进行逐个判断
        while(rotateArray[left] >= rotateArray[right]){
            if(right-left==1){
                mid = right;
                break;
            }
            mid = (left+right)/2;
            if(rotateArray[mid]==rotateArray[left] && rotateArray[mid]==rotateArray[right]){
                for(int i=left;i<right;i++){
                    if(rotateArray[i]>rotateArray[i+1]){
                        mid = i;
                        return rotateArray[mid];
                    }
                }
            }
            else if(rotateArray[mid]>=rotateArray[left]){
                left = mid;
            }
            else if(rotateArray[mid]<=rotateArray[right]){
                right = mid;
            }
        }
        return rotateArray[mid];
    }
};