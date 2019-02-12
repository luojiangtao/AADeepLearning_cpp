#ifndef  __LAYER_HPP__
#define __LAYER_HPP__
#include <iostream>
#include <memory>
#include "Blob.hpp"

using std::vector;
using namespace std;
using std::shared_ptr;

struct Param  //结构体 （层参数，主要是每一层的细节部分）
{
	/*1.卷积层超参数 */
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;

	/*2.池化层超参数*/
	int pool_stride;
	int pool_width;
	int pool_height;

	/*3.全连接层超参数（即该层神经元个数） */
	int fc_kernels;
};

//父类，方便多态调用
class Layer
{
public:
	Layer(){};
	~Layer(){};
	virtual void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) = 0;
	virtual void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param) = 0;
	virtual void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param) = 0;

	virtual void backward(const shared_ptr<Blob>& din,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads,
		const Param& param) = 0;
};

//卷积层
class ConvLayer:public Layer
{
public:
	ConvLayer(){};
	~ConvLayer(){};
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param);
	void backward(const shared_ptr<Blob>& din,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads,
		const Param& param);
};

//激活
class ReluLayer :public Layer
{
public:
	ReluLayer(){};
	~ReluLayer(){};
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param);
	void backward(const shared_ptr<Blob>& din,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads,
		const Param& param);
};
//池化
class PoolLayer :public Layer
{
public:
	PoolLayer(){};
	~PoolLayer(){};
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param);
	void backward(const shared_ptr<Blob>& din,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads,
		const Param& param);
};
//全连接层
class FcLayer :public Layer
{
public:
	FcLayer(){};
	~FcLayer(){};
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param);
	void backward(const shared_ptr<Blob>& din,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads,
		const Param& param);
};
//交叉熵损失层
class SoftmaxLossLayer
{
public:
	static void softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout);
};

#endif  //__LAYER_HPP__