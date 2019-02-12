#include "Blob.hpp"
#include "cassert"

using namespace std;
using namespace arma;

//构造函数
Blob::Blob(const int batch_size, const int channel, const int height, const int width, int type) : batch_size(batch_size), channel(channel), height(height), width(width)
{
	//arma_rng::set_seed_random();  //系统随机生成种子(如果没有这一句，就会每次启动程序(进程)时都默认从种子1开始来生成随机数！
	_init(batch_size, channel, height, width, type);
}
Blob::Blob(const vector<int> shape_, int type) : batch_size(shape_[0]), channel(shape_[1]), height(shape_[2]), width(shape_[3])
{
	//arma_rng::set_seed_random();  //系统随机生成种子(如果没有这一句，就会每次启动程序(进程)时都默认从种子1开始来生成随机数！
	_init(batch_size, channel, height, width, type);
}
//初始化
void Blob::_init(const int batch_size, const int channel, const int height, const int width, int type)
{
	if (type == TZEROS){
		blob_data = vector<cube>(batch_size, cube(height, width, channel, fill::zeros));
	}
	else if (type == TONES){
		blob_data = vector<cube>(batch_size, cube(height, width, channel, fill::ones));
	}
	else if (type == TRANDU){
		for (int i = 0; i < batch_size; ++i){
			blob_data.push_back(arma::randu<cube>(height, width, channel));//堆叠
		}
	}
	else if (type == TRANDN){
		for (int i = 0; i < batch_size; ++i){
			blob_data.push_back(arma::randn<cube>(height, width, channel));//堆叠
		}
	}
	else{
		blob_data = vector<cube>(batch_size, cube(height, width, channel));
	}
}

vector<int> Blob::size() const
{
	vector<int> shape_{ batch_size,
		channel,
		height,
		width };
	return shape_;
}
//打印blob里面的数据
void Blob::print(string str)
{
	assert(!blob_data.empty()); //断言：   blob_data不为空！否则中止程序
	cout << str << endl;
	for (int i = 0; i < batch_size; i++)
	{
		printf("N = %d\n", i); //N_为blob_data中cube个数
		this->blob_data[i].print();//逐一打印cube，调用cube中重载好的print()
	}

}
//重载运算符
cube& Blob::operator[] (int i)
{
	return blob_data[i];
}

//重载运算符
Blob& Blob::operator*= (const double k)
{
	for (int i = 0; i < batch_size; i++)
	{
		blob_data[i] = blob_data[i] * k;   //调用cube中实现的*操作符
	}
	return *this;
}
//重载运算符
Blob& Blob::operator= (double val)
{
	for (int i = 0; i < batch_size; ++i)
	{
		blob_data[i].fill(val);   //调用cube中实现的*操作符
	}
	return *this;
}

Blob operator*(Blob& A, Blob& B)  //友元函数的具体实现：这里没有类限定例如 (Blob& Blob::)这种形式
{
	//(1). 确保两个输入Blob尺寸一样
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_A[i] == size_B[i]);  //断言：两个输入Blob的尺寸（N,C,H,W）一样！
	}
	//(2). 遍历所有的cube，每一个cube做对应位置相乘（cube % cube）
	int N = size_A[0];
	Blob C(A.size());
	for (int i = 0; i < N; ++i)
	{
		C[i] = A[i] % B[i];
	}
	return C;
}

Blob operator*(double num, Blob& B)
{
	//遍历所有的cube，每一个cube都乘上一个数值num
	int N = B.get_batch_size();
	Blob out(B.size());
	for (int i = 0; i < N; ++i)
	{
		out[i] = num * B[i];
	}
	return out;
}

Blob operator+(Blob& A, Blob& B)
{
	//(1). 确保两个输入Blob尺寸一样
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_A[i] == size_B[i]);  //断言：两个输入Blob的尺寸（N,C,H,W）一样！
	}
	//(2). 遍历所有的cube，每一个cube做对应位置相加（cube + cube）
	int N = size_A[0];
	Blob C(A.size());
	for (int i = 0; i < N; ++i)
	{
		C[i] = A[i] + B[i];
	}
	return C;
}

vector<cube>& Blob::get_data()
{
	return blob_data;
}

Blob Blob::subBlob(int low_idx, int high_idx)
{
	//举例： [0,1,2,3,4,5]  -> [1,3)  -> [1,2]
	if (high_idx > low_idx)
	{
		Blob tmp(high_idx - low_idx, channel, height, width);  // high_idx > low_idx
		for (int i = low_idx; i < high_idx; ++i)
		{
			tmp[i - low_idx] = (*this)[i];
		}
		return tmp;
	}
	else
	{
		// low_idx >high_idx
		//举例： [0,1,2,3,4,5]  -> [3,2)-> (6 - 3) + (2 -0) -> [3,4,5,0]
		Blob tmp(batch_size - low_idx + high_idx, channel, height, width);
		for (int i = low_idx; i < batch_size; ++i)   //分开两段截取：先截取第一段
		{
			tmp[i - low_idx] = (*this)[i];
		}
		for (int i = 0; i < high_idx; ++i)   //分开两段截取：再截取循环到从0开始的这段
		{
			tmp[i + batch_size - low_idx] = (*this)[i];
		}
		return tmp;
	}
}


Blob Blob::pad(int pad, double val)
{
	assert(!blob_data.empty());
	Blob padX(batch_size, channel, height + 2 * pad, width + 2 * pad);

	for (int n = 0; n < batch_size; ++n)
	{
		for (int c = 0; c < channel; ++c)
		{
			for (int h = 0; h < height; ++h)
			{
				for (int w = 0; w < width; ++w)
				{
					padX[n](h + pad, w + pad, c) = blob_data[n](h, w, c);
				}
			}
		}
	}
	return padX;
}
Blob Blob::deletePad(int pad)
{
	assert(!blob_data.empty());   //断言：Blob自身不为空
	Blob out(batch_size, channel, height - 2 * pad, width - 2 * pad);
	for (int n = 0; n < batch_size; ++n)
	{
		for (int c = 0; c < channel; ++c)
		{
			for (int h = pad; h < height - pad; ++h)
			{
				for (int w = pad; w < width - pad; ++w)
				{
					//注意，out的索引是从0开始的，所以要减去pad
					out[n](h - pad, w - pad, c) = blob_data[n](h, w, c);
				}
			}
		}
	}
	return out;
}
void Blob::maxIn(double val)
{
	assert(!blob_data.empty());
	for (int i = 0; i < batch_size; ++i)
	{
		/*
		.transform(lambda_function) (C++11 Only)  这个方法只支持c++11以上版本，现在编译器基本都支持c++11了
		传入一个lambda函数，实现你所要的变换功能！
		适用于Mat, Col, Row和Cube；对于矩阵，按照column-by-column来进行变换；
		对于立方体，按照slice-by-slice进行变换，每一个slice是一个矩阵。
		*/
		blob_data[i].transform([val](double e){return e>val ? e : val; });
	}
	return;
}