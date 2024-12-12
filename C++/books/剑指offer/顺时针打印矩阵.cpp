/*
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
*/
#include <iostream>
#include <vector>
using namespace std;

class Solution{
public:
    void GetNum(int start, vector<vector<int> > matrix, vector<int>& out){
        // 顺时针打印一圈
        int endx = matrix[0].size()-start-1;
        int endy = matrix.size()-start-1;

        // 打印最上面一行
        for(int i=start;i<=endx;i++){
            out.push_back(matrix[start][i]);
        }   

        // 打印最右边一行
        if(endy>start){
            for(int j=start+1;j<=endy;j++)
                out.push_back(matrix[j][endx]);
        }

        // 打印最下面一行
        if(endx>start){
            for(int i=endx-1;i>=start;i--)
                out.push_back(matrix[endy][i]);
        }

        // 打印最左边一行
        if(endy>start){
            for(int j=endy-1;j>start;j--)
                out.push_back(matrix[j][start]);
        }
    }

    vector<int> printMatrix(vector<vector<int> > matrix){
        // 逐层打印外围圈
        if(!matrix.size() || !matrix[0].size())
            return {};

        vector<int> out;
        int rows = matrix.size();
        int cols = matrix[0].size();
        int start = 0;
        while(start <= (rows-1)/2 && start <= (cols-1)/2){
            GetNum(start, matrix, out);
            start++;
        }
        return out;
    } 
};

int main(){
    Solution sol;
    vector<vector<int> > matrix = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    vector<int> out = sol.printMatrix(matrix);
    for(int num:out)
        cout<<num<<" ";
    cout<<endl;
    return 0;
}