/*
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
*/
class Solution {
public:
	//每次得到最外围圈数字
    void GetNum(int start, vector<vector<int>> nums, vector<int>& out){
        int endx = nums[0].size()-start-1;
        int endy = nums.size()-start-1;
        for(int i=start;i<=endx;i++){
            out.push_back(nums[start][i]);
        }
        if(endy>start){
            for(int j=start+1;j<=endy;j++)
                out.push_back(nums[j][endx]);
            if(endx>start)
                for(int i=endx-1;i>=start;i--)
                    out.push_back(nums[endy][i]);
            if(endx>start)
                for(int j=endy-1;j>start;j--)
                    out.push_back(nums[j][start]);
        }
        return;
    }
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> out;
        if(!matrix.size() || !matrix[0].size())
            return out;
        int rows = matrix.size();
        int cols = matrix[0].size();
        int start = 0;
        while(start <= (rows-1)/2 && start <= (cols-1)/2){
            GetNum(start,matrix,out);
            start++;
        }
        return out;
    }
};