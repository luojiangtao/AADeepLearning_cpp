#include "Layer.hpp"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace arma;
void ConvLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	int kernel_number = param.conv_kernels;
	int channel = inShape[1];
	int height = param.conv_height;
	int width = param.conv_width;

	//2.初始化存储W和b的Blob  (in[1]->W和in[2]->b)
	if (!in[1])   //存储W的Blob不为空
	{
		in[1].reset(new Blob(kernel_number, channel, height, width, TRANDN)); //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
		//(*in[1]) *= 1e-2;
		cout << "initLayer: " << lname << "  Init weights  with standard Gaussian ;" << endl;
	}

	//2.初始化存储W和b的Blob  (in[1]->W和in[2]->b)
	if (!in[2])   //存储W的Blob不为空
	{
		in[2].reset(new Blob(kernel_number, 1, 1, 1, TRANDN)); //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
		//(*in[2]) *= 1e-2;
		cout << "initLayer: " << lname << "  Init bias  with standard Gaussian ;" << endl;
	}
}

void ConvLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param)
{
	int batch_size = inShape[0];
	int channel = inShape[1];
	int height = inShape[2];
	int width = inShape[3];

	int kernel_number = param.conv_kernels;
	int conv_height = param.conv_height;
	int conv_width = param.conv_width;
	int conv_pad = param.conv_pad;
	int conv_stride = param.conv_stride;

	outShape[0] = batch_size; //输出样本数
	outShape[1] = kernel_number; //输出通道数
	outShape[2] = (height + 2 * conv_pad - conv_height) / conv_stride + 1; //输出高
	outShape[3] = (width + 2 * conv_pad - conv_width) / conv_stride + 1; //输出宽
}

template<typename T>
void Arma_mat2cv_mat(const arma::Mat<T>& arma_mat_in, cv::Mat_<T>& cv_mat_out)  //将arma::Mat转变为cv::Mat
{
	cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat_in.n_cols),
		static_cast<int>(arma_mat_in.n_rows),
		const_cast<T*>(arma_mat_in.memptr())), cv_mat_out);
	return;
};

void visiable(const cube& in, vector<cv::Mat_<double>>& vec_mat)   //可视化一个cube中的所有Mat
{
	int num = in.n_slices;
	for (int i = 0; i < num; ++i)
	{
		cv::Mat_<double> mat_cv;
		arma::mat mat_arma = in.slice(i);
		Arma_mat2cv_mat<double>(mat_arma, mat_cv);
		vec_mat.push_back(mat_cv);
	}
	return;
}

void ConvLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param)
{
	if (out)
		out.reset();
	//-------step1.获取相关尺寸（输入，卷积核，输出）
	assert(in[0]->get_channel() == in[1]->get_channel());  //断言：输入Blob通道数和卷积核Blob通道数一样（务必保证这一点）

	int N = in[0]->get_batch_size();        //输入Blob中cube个数（该batch样本个数）
	int C = in[0]->get_channel();         //输入Blob通道数
	int Hx = in[0]->get_height();      //输入Blob高
	int Wx = in[0]->get_width();    //输入Blob宽

	int F = in[1]->get_batch_size();		  //卷积核个数
	int Hw = in[1]->get_height();     //卷积核高
	int Ww = in[1]->get_width();   //卷积核宽

	int Ho = (Hx + param.conv_pad * 2 - Hw) / param.conv_stride + 1;    //输出Blob高（卷积后）
	int Wo = (Wx + param.conv_pad * 2 - Ww) / param.conv_stride + 1;  //输出Blob宽（卷积后）
	//-------step2.根据要求做padding操作
	Blob padX = in[0]->pad(param.conv_pad);
	//padX[0].slice(0).print("pad=");
	//arma::mat mat0_arma = padX[0].slice(0);
	//cv::Mat_<double> mat0_cv;
	//Arma_mat2cv_mat(mat0_arma, mat0_cv);
	//-------step3.开始卷积运算
	out.reset(new Blob(N, F, Ho, Wo));
	for (int n = 0; n < N; ++n)   //输出cube数
	{
		for (int f = 0; f < F; ++f)  //输出通道数
		{
			for (int hh = 0; hh < Ho; ++hh)   //输出Blob的高
			{
				for (int ww = 0; ww < Wo; ++ww)   //输出Blob的宽
				{
					cube window = padX[n](span(hh*param.conv_stride, hh*param.conv_stride + Hw - 1),
						span(ww*param.conv_stride, ww*param.conv_stride + Ww - 1),
						span::all);
					//out = Wx+b
					(*out)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);    //b = (F,1,1,1)
				}
			}
		}
	}
	//(*in[1])[0].print("W=");
	//cout << "b=\n" << as_scalar((*in[2])[0]) << "\n" << endl;
	//(*out)[0].slice(0).print("out=");

	//vector < cv::Mat_<double>> vec_mat_w1;
	//visiable((*in[1])[0], vec_mat_w1);    //可视化第一个卷积核

	//vector < cv::Mat_<double>> vec_mat_b1;
	//visiable((*in[2])[0], vec_mat_b1);    //可视化第一个偏置核

	//vector < cv::Mat_<double>> vec_mat_out;
	//visiable((*out)[0], vec_mat_out);    //可视化第一个输出cube


	return;
}

void ConvLayer::backward(const shared_ptr<Blob>& din,   //输入梯度
	const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads,
	const Param& param)
{
	//step1. 设置输出梯度Blob的尺寸（dX---grads[0]）
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));
	//step2. 获取输入梯度Blob的尺寸（din）
	int Nd = din->get_batch_size();        //输入梯度Blob中cube个数（该batch样本个数）
	int Cd = din->get_channel();         //输入梯度Blob通道数
	int Hd = din->get_height();      //输入梯度Blob高
	int Wd = din->get_width();    //输入梯度Blob宽
	//step3. 获取卷积核相关参数
	int Hw = param.conv_height;
	int Ww = param.conv_width;
	int stride = param.conv_stride;

	//step4. 填充操作
	Blob pad_X = cache[0]->pad(param.conv_pad);  //参与实际反向传播计算的应该是填充过的特征Blob
	Blob pad_dX(pad_X.size(), TZEROS);                      //梯度Blob应该与该层的特征Blob尺寸保持一致

	//step5. 开始反向传播
	for (int n = 0; n < Nd; ++n)   //遍历输入梯度din的样本数
	{
		for (int c = 0; c < Cd; ++c)  //遍历输入梯度din的通道数
		{
			for (int hh = 0; hh < Hd; ++hh)   //遍历输入梯度din的高
			{
				for (int ww = 0; ww < Wd; ++ww)   //遍历输入梯度din的宽
				{
					//(1). 通过滑动窗口，截取不同输入特征片段
					cube window = pad_X[n](span(hh*stride, hh*stride + Hw - 1), span(ww*stride, ww*stride + Ww - 1), span::all);
					//(2). 计算梯度
					//dX
					pad_dX[n](span(hh*stride, hh*stride + Hw - 1), span(ww*stride, ww*stride + Ww - 1), span::all) += (*din)[n](hh, ww, c) * (*cache[1])[c];
					//dW  --->grads[1]
					(*grads[1])[c] += (*din)[n](hh, ww, c) * window / Nd;
					//db   --->grads[2]
					(*grads[2])[c](0, 0, 0) += (*din)[n](hh, ww, c) / Nd;
				}
			}
		}
	}

	//step6. 去掉输出梯度中的padding部分
	(*grads[0]) = pad_dX.deletePad(param.conv_pad);

	////测试代码
	//(*din)[0].slice(0).print("input:   din=");				    //输入梯度：打印第一个din的第一个矩阵
	//(*din)[0].slice(1).print("input:   din=");				    //输入梯度：打印第一个din的第二个矩阵
	//(*din)[0].slice(2).print("input:   din=");				    //输入梯度：打印第一个din的第三个矩阵
	//(*cache[1])[0].slice(0).print("W1=");		                //打印第一个卷积核的第一个矩阵	
	//(*cache[1])[1].slice(0).print("W2=");		                //打印第二个卷积核的第一个矩阵	
	//(*cache[1])[2].slice(0).print("W3=");		                //打印第三个卷积核的第一个矩阵		
	//pad_dX[0].slice(0).print("output:   pad_dX=");		//输出梯度：打印第一个pad_dX的第一个矩阵

	return;
}


void ReluLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	cout << "ReluLayer::initLayer " << endl;
}
void ReluLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param)
{
	//尺寸不变，直接复制
	outShape.assign(inShape.begin(), inShape.end());
}
void ReluLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param)
{
	if (out)
	{
		out.reset();
	}
	out.reset(new Blob(*in[0]));
	out->maxIn(0);
}
void ReluLayer::backward(const shared_ptr<Blob>& din,
	const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads,
	const Param& param)
{
	//step1. 设置输出梯度Blob的尺寸（dX---grads[0]）
	grads[0].reset(new Blob(*cache[0]));

	//step2. 获取掩码mask
	int N = grads[0]->get_batch_size();
	for (int n = 0; n < N; ++n)
	{
		(*grads[0])[n].transform([](double e) {return e > 0 ? 1 : 0; });
	}
	(*grads[0]) = (*grads[0]) * (*din);

	//(*din)[0].slice(0).print("din=");				//输入梯度：打印第一个din的第一个矩阵
	//(*cache[0])[0].slice(0).print("cache=");		//掩码： 打印第一个cache的第一个矩阵
	//(*grads[0])[0].slice(0).print("grads=");		//输出梯度：打印第一个grads的第一个矩阵
	return;
}
void PoolLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	cout << "PoolLayer::initLayer " << endl;
}
void PoolLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param)
{
	int batch_size = inShape[0];
	int channel = inShape[1];
	int height = inShape[2];
	int width = inShape[3];

	int pool_height = param.pool_height;
	int pool_width = param.pool_width;
	int pool_stride = param.pool_stride;

	outShape[0] = batch_size; //输出样本数
	outShape[1] = channel; //输出通道数
	outShape[2] = (height - pool_height) / pool_stride + 1; //输出高
	outShape[3] = (width - pool_width) / pool_stride + 1; //输出宽
}
void PoolLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param)
{
	if (out)
		out.reset();
	//-------step1.获取相关尺寸（输入，池化核，输出）
	int N = in[0]->get_batch_size();        //输入Blob中cube个数（该batch样本个数）
	int C = in[0]->get_channel();         //输入Blob通道数
	int Hx = in[0]->get_height();      //输入Blob高
	int Wx = in[0]->get_width();    //输入Blob宽

	int Hw = param.pool_height;     //池化核高
	int Ww = param.pool_width;   //池化核宽

	int Ho = (Hx - Hw) / param.pool_stride + 1;    //输出Blob高（池化后）
	int Wo = (Wx - Ww) / param.pool_stride + 1;  //输出Blob宽（池化后）

	//-------step2.开始池化
	out.reset(new Blob(N, C, Ho, Wo));

	for (int n = 0; n < N; ++n)   //输出cube数
	{
		for (int c = 0; c < C; ++c)  //输出通道数
		{
			for (int hh = 0; hh < Ho; ++hh)   //输出Blob的高
			{
				for (int ww = 0; ww < Wo; ++ww)   //输出Blob的宽
				{
					(*out)[n](hh, ww, c) = (*in[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hw - 1),
						span(ww*param.pool_stride, ww*param.pool_stride + Ww - 1),
						span(c, c)).max();
				}
			}
		}
	}
	//vector < cv::Mat_<double>> vec_mat_in;
	//visiable((*in[0])[0], vec_mat_in);    //可视化经过relu后的第一个输出cube
	//vector < cv::Mat_<double>> vec_mat_out;
	//visiable((*out)[0], vec_mat_out);    //可视化经过relu后的第一个输出cube
}

void PoolLayer::backward(const shared_ptr<Blob>& din,
	const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads,
	const Param& param)
{
	//step1. 设置输出梯度Blob的尺寸（dX---grads[0]）
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	//step2. 获取输入梯度Blob的尺寸（din）
	int Nd = din->get_batch_size();        //输入梯度Blob中cube个数（该batch样本个数）
	int Cd = din->get_channel();         //输入梯度Blob通道数
	int Hd = din->get_height();      //输入梯度Blob高
	int Wd = din->get_width();    //输入梯度Blob宽

	//step3. 获取池化核相关参数
	int Hp = param.pool_height;
	int Wp = param.pool_width;
	int stride = param.pool_stride;

	//step4. 开始反向传播
	for (int n = 0; n < Nd; ++n)   //输出cube数
	{
		for (int c = 0; c < Cd; ++c)  //输出通道数
		{
			for (int hh = 0; hh < Hd; ++hh)   //输出Blob的高
			{
				for (int ww = 0; ww < Wd; ++ww)   //输出Blob的宽
				{
					//(1). 获取掩码mask
					mat window = (*cache[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hp - 1),
						span(ww*param.pool_stride, ww*param.pool_stride + Wp - 1),
						span(c, c));
					double maxv = window.max();
					mat mask = conv_to<mat>::from(maxv == window);  //"=="返回的是一个umat类型的矩阵！umat转换为mat
					//(2). 计算梯度
					(*grads[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hp - 1),
						span(ww*param.pool_stride, ww*param.pool_stride + Wp - 1),
						span(c, c)) += mask*(*din)[n](hh, ww, c);  //umat  -/-> mat
				}
			}
		}
	}
	//(*din)[0].slice(0).print("din=");
	//(*cache[0])[0].slice(0).print("cache=");  //mask
	//(*grads[0])[0].slice(0).print("grads=");

	return;
}

void FcLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param)
{
	int fc_kernels = param.fc_kernels;
	int channel = inShape[1];
	int height = inShape[2];
	int width = inShape[3];

	//2.初始化存储W和b的Blob  (in[1]->W和in[2]->b)
	if (!in[1])   //存储W的Blob不为空
	{
		in[1].reset(new Blob(fc_kernels, channel, height, width, TRANDN)); //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
		(*in[1]) *= 1e-2;
		cout << "initLayer: " << lname << "  Init weights  with standard Gaussian ;" << endl;
	}

	//2.初始化存储W和b的Blob  (in[1]->W和in[2]->b)
	if (!in[2])   //存储W的Blob不为空
	{
		in[2].reset(new Blob(fc_kernels, 1, 1, 1, TRANDN)); //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
		(*in[2]) *= 1e-2;
		cout << "initLayer: " << lname << "  Init bias  with standard Gaussian ;" << endl;
	}
}
void FcLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const Param& param)
{
	int batch_size = inShape[0];

	//（200,10,1,1）
	outShape[0] = batch_size; //输出样本数
	outShape[1] = param.fc_kernels; //输出通道数，也就是神经元个数
	outShape[2] = 1; //输出高
	outShape[3] = 1; //输出宽
}
void FcLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param)
{
	if (out)
		out.reset();
	//-------step1.获取相关尺寸（输入，全连接核，输出）
	int N = in[0]->get_batch_size();        //输入Blob中cube个数（该batch样本个数）
	int C = in[0]->get_channel();         //输入Blob通道数
	int Hx = in[0]->get_height();      //输入Blob高
	int Wx = in[0]->get_width();    //输入Blob宽

	int F = in[1]->get_batch_size();		  //全连接核个数
	int Hw = in[1]->get_height();     //全连接核高
	int Ww = in[1]->get_width();   //全连接核宽
	assert(in[0]->get_channel() == in[1]->get_channel());  //断言：输入Blob通道数和全连接核Blob通道数一样（务必保证这一点）
	assert(Hx == Hw  && Wx == Ww);  //断言：输入Blob高和宽和全连接核Blob高和宽一样（务必保证这一点）

	int Ho = 1;    //输出Blob高（全连接操作后）
	int Wo = 1;  //输出Blob宽（全连接操作后）

	//-------step2.开始全连接运算
	out.reset(new Blob(N, F, Ho, Wo));

	for (int n = 0; n < N; ++n)   //输出cube数
	{
		for (int f = 0; f < F; ++f)  //输出通道数
		{
			(*out)[n](0, 0, f) = accu((*in[0])[n] % (*in[1])[f]) + as_scalar((*in[2])[f]);    //b = (F,1,1,1)
		}
	}

	//vector < cv::Mat_<double>> vec_mat_CUBE1;
	//visiable((*in[0])[0], vec_mat_CUBE1);    //可视化第一个全连接核

	//vector < cv::Mat_<double>> vec_mat_w1;
	//visiable((*in[1])[0], vec_mat_w1);    //可视化第一个全连接核

	//vector < cv::Mat_<double>> vec_mat_b1;
	//visiable((*in[2])[0], vec_mat_b1);    //可视化第一个偏置核

	//vector < cv::Mat_<double>> vec_mat_out;
	//visiable((*out)[0], vec_mat_out);    //可视化第一个输出cube


}

void FcLayer::backward(const shared_ptr<Blob>& din,
	const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads,
	const Param& param)
{
	//shared_ptr<Blob> din(new Blob(2, 2, 1, 1, TRANDU));
	//vector<shared_ptr<Blob>> cache(3, NULL);
	//cache[0].reset(new Blob(2, 2, 2, 2, TONES));
	//cache[1].reset(new Blob(2, 2, 2, 2, TRANDU));
	//cache[2].reset(new Blob(2, 1, 1, 1, TRANDU));

	//dX,dW,db  -> X,W,b
	grads[0].reset(new Blob(cache[0]->size(), TZEROS));
	grads[1].reset(new Blob(cache[1]->size(), TZEROS));
	grads[2].reset(new Blob(cache[2]->size(), TZEROS));
	int N = grads[0]->get_batch_size();
	int F = grads[1]->get_batch_size();
	assert(F == cache[1]->get_batch_size());

	for (int n = 0; n < N; ++n)
	{
		for (int f = 0; f < F; ++f)
		{
			//dX
			(*grads[0])[n] += (*din)[n](0, 0, f) * (*cache[1])[f];
			//dW
			(*grads[1])[f] += (*din)[n](0, 0, f) * (*cache[0])[n] / N;
			//db
			(*grads[2])[f] += (*din)[n](0, 0, f) / N;
		}
	}

	//vector < cv::Mat_<double>> vec_mat_dout0;
	//visiable((*din)[0], vec_mat_dout0);      //可视化第一个误差信号（后层传来的梯度）

	//vector < cv::Mat_<double>> vec_mat_dout1;
	//visiable((*din)[1], vec_mat_dout1);      //可视化第一个误差信号（后层传来的梯度）

	//vector < cv::Mat_<double>> vec_mat_x0;
	//visiable((*cache[0])[0], vec_mat_x0);     //可视化第一个特征cube

	//vector < cv::Mat_<double>> vec_mat_x1;
	//visiable((*cache[0])[1], vec_mat_x1);     //可视化第二个特征cube

	//vector < cv::Mat_<double>> vec_mat_w0;
	//visiable((*cache[1])[0], vec_mat_w0);    //可视化第一个全连接核

	//vector < cv::Mat_<double>> vec_mat_w1;
	//visiable((*cache[1])[1], vec_mat_w1);    //可视化第二个全连接核

	//vector < cv::Mat_<double>> vec_mat_dx0;
	//visiable((*grads[0])[0], vec_mat_dx0);     //可视化第一个dx

	//vector < cv::Mat_<double>> vec_mat_dw0;
	//visiable((*grads[1])[0], vec_mat_dw0);    //可视化第一个dw

	//vector < cv::Mat_<double>> vec_mat_db0;
	//visiable((*grads[2])[0], vec_mat_db0);    //可视化第一个db
	return;
}



void SoftmaxLossLayer::softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout)
{
	if (dout)
		dout.reset();
	//-------step1.获取相关尺寸
	int N = in[0]->get_batch_size();        //输入Blob中cube个数（该batch样本个数）
	int C = in[0]->get_channel();         //输入Blob通道数
	int Hx = in[0]->get_height();      //输入Blob高
	int Wx = in[0]->get_width();    //输入Blob宽
	assert(Hx == 1 && Wx == 1);

	dout.reset(new Blob(N, C, Hx, Wx));   //（N,C,1,1）
	double loss_ = 0;
	//(*in[0])[0].print();
	//system("pause");
	for (int i = 0; i < N; ++i)
	{
		cube prob = arma::exp((*in[0])[i]) / arma::accu(arma::exp((*in[0])[i]));    //softmax归一化
		loss_ += (-arma::accu((*in[1])[i] % arma::log(prob)));  //累加各个样本的交叉熵损失值
		//梯度表达式推导：https://blog.csdn.net/qian99/article/details/78046329
		(*dout)[i] = prob - (*in[1])[i];  //计算各个样本产生的误差信号（反向梯度）
	}
	loss = loss_ / N;   //求平均损失

	return;
}