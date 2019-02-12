#ifndef  __BLOB_HPP__
#define __BLOB_HPP__
#include <vector>
#include <armadillo>
using std::vector;
using std::string;
using arma::cube;

enum FillType
{
	TONES = 1,  //cube所有元素都填充为1
	TZEROS = 2, //cube所有元素都填充为0
	TRANDU = 3,  //将元素设置为[0,1]区间内均匀分布的随机值
	TRANDN = 4,  //使用μ= 0和σ= 1的标准高斯分布设置元素
	TDEFAULT = 5
};
//矩阵数据存储和计算
class Blob
{
public:
	//batch_size(0), channel(0), height(0), width(0) 都初始化为0
	Blob() : batch_size(0), channel(0), height(0), width(0)
	{};
	Blob(const int batch_size, const int channel, const int height, const int width, int type = TDEFAULT);
	Blob(const vector<int> shape_, int type = TDEFAULT);  //重载函数

	void print(string str = "");
	vector<cube>& get_data();
	cube& operator[] (int i);
	Blob& operator= (double val);
	Blob& operator *= (const double k);
	friend Blob operator*(Blob& A, Blob& B);  //声明为友元函数
	friend Blob operator*(double num, Blob& B);  //声明为友元函数
	friend Blob operator+(Blob& A, Blob& B);  //声明为友元函数
	Blob subBlob(int low_idx, int high_idx);
	Blob pad(int pad, double val = 0.0);
	Blob deletePad(int pad);
	void maxIn(double val = 0.0);
	vector<int> size() const;

	inline int get_batch_size() const
	{
		return batch_size;
	}
	inline int get_channel() const
	{
		return channel;
	}
	inline int get_height() const
	{
		return height;
	}
	inline int get_width() const
	{
		return width;
	}

private:
	void _init(const int batch_size, const int channel, const int height, const int width, int type);
private:
	int batch_size; //cube个数，也是样本数
	int channel; // 通道数
	int height; // 高
	int width; // 宽
	vector<cube> blob_data; // 多个cube
};
#endif  //__BLOB_HPP__