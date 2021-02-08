/*
请实现一个函数，将一个字符串中的每个空格替换成“%20”。
例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
*/
class Solution {
public:
	void replaceSpace(char *str,int length) {
		//创建一个新的数组即可
	    string new_str;
	    for (int i = 0; i < length; i++){
		    if (str[i] == ' '){
			    new_str.insert(new_str.end(),'%');
			    new_str.insert(new_str.end(), '2');
			    new_str.insert(new_str.end(), '0');
		    }
		    else{
			    new_str.insert(new_str.end(), str[i]);
		    }
	    }
	    new_str.insert(new_str.end(), '\0');
	    for (int i = 0; i < new_str.size(); i++){
		    str[i] = new_str[i];
	    }
	}
};