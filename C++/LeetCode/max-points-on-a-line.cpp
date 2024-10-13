/**
 * Definition for a point.
 * struct Point {
 *     int x;
 *     int y;
 *     Point() : x(0), y(0) {}
 *     Point(int a, int b) : x(a), y(b) {}
 * };
 */
#include <algorithm>
int gcd(int a, int b){
    int r;
    while(b){
        r = a%b;
        a = b;
        b = r;
    }
    return a;
}
bool cmp(Point p1, Point p2){
    if(p1.x==p2.x)
        return p1.y<p2.y;
    return p1.x<p2.x;
}
class Solution {
public:
    int maxPoints(vector<Point> &points) {
        if(points.size()<=1)
            return points.size();
        sort(points.begin(),points.end(),cmp);
        map<pair<int, int>,int> ks;
        int ans = 0;
        for(int i=0;i<points.size();i++){
            int mul=0,k_num=0,ver=0;
            for(int j=i;j<points.size();j++){
                int di_x = points[i].x - points[j].x;
                int di_y = points[i].y - points[j].y;
                if(!di_x && !di_y)
                    mul++;
                else if(!di_x)
                    ver++;
                else{
                    int com = gcd(di_x, di_y);
                    di_x /= com;
                    di_y /= com;
                    if(!ks[pair<int,int>(di_x,di_y)])
                        ks[pair<int,int>(di_x,di_y)] = 1;
                    else
                        ks[pair<int,int>(di_x,di_y)]++;
                }
                if(k_num < ks[pair<int,int>(di_x,di_y)])
                    k_num = ks[pair<int,int>(di_x,di_y)];
                if(k_num < ver)
                    k_num = ver;
            }
            if(mul + ver > ans)
                ans = mul + ver;
            if(k_num + mul > ans)
                ans = mul + k_num;
            ks.clear();
        }
        return ans;
    }
};