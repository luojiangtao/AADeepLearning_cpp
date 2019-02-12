#include "Net.hpp"
#include "Blob.hpp"
#include <json/json.h>
#include <fstream>
#include <cassert>
#include <memory>

#include<windows.h>

using namespace std;//使用std命名空间


void NetParam::readNetParam(string file)
{
	ifstream ifs;
	ifs.open(file);
	assert(ifs.is_open());   //断言：确保json文件正确打开
	Json::Reader reader;  //  解析器
	Json::Value value;      //存储器
	if (reader.parse(ifs, value))
	{
		if (!value["train"].isNull())
		{
			//引用就是别名的意思
			auto &tparam = value["train"];  //通过引用方式，可以拿到“train”对象里面的所有元素
			this->lr = tparam["learning rate"].asDouble(); //解析成Double类型存放
			this->lr_decay = tparam["lr decay"].asDouble();
			//this->update = tparam["update method"].asString();//解析成String类型存放
			this->optimizer = tparam["optimizer"].asString();//解析成String类型存放
			this->momentum = tparam["momentum parameter"].asDouble();
			this->num_epochs = tparam["num epochs"].asInt();//解析成Int类型存放
			this->use_batch = tparam["use batch"].asBool();//解析成Bool类型存放
			this->batch_size = tparam["batch size"].asInt();
			this->eval_interval = tparam["evaluate interval"].asInt();
			this->lr_update = tparam["lr update"].asBool();
			this->snap_shot = tparam["snapshot"].asBool();
			this->snapshot_interval = tparam["snapshot interval"].asInt();
			this->fine_tune = tparam["fine tune"].asBool();
			this->preTrainModel = tparam["pre train model"].asString();//解析成String类型存放
		}
		if (!value["net"].isNull())
		{
			auto &nparam = value["net"];                                //通过引用方式，拿到“net”数组里面的所有对象
			for (int i = 0; i < (int)nparam.size(); ++i)                //遍历“net”数组里面的所有对象
			{
				auto &ii = nparam[i];                                          //通过引用方式，拿到当前对象里面的所有元素
				this->layers.push_back(ii["name"].asString());  //往层名vector中堆叠名称      a=[]   a.append()
				this->ltypes.push_back(ii["type"].asString());   //往层类型vector中堆叠类型

				if (ii["type"].asString() == "Conv")
				{
					int num = ii["kernel num"].asInt();
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					int pad = ii["pad"].asInt();
					int stride = ii["stride"].asInt();

					this->lparams[ii["name"].asString()].conv_stride = stride;
					this->lparams[ii["name"].asString()].conv_kernels = num;
					this->lparams[ii["name"].asString()].conv_pad = pad;
					this->lparams[ii["name"].asString()].conv_width = width;
					this->lparams[ii["name"].asString()].conv_height = height;
				}
				if (ii["type"].asString() == "Pool")
				{
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					int stride = ii["stride"].asInt();
					this->lparams[ii["name"].asString()].pool_stride = stride;
					this->lparams[ii["name"].asString()].pool_width = width;
					this->lparams[ii["name"].asString()].pool_height = height;
				}
				if (ii["type"].asString() == "Fc")
				{
					int num = ii["kernel num"].asInt();
					this->lparams[ii["name"].asString()].fc_kernels = num;
				}
			}
		}


	}

};

void Net::init(NetParam& param, vector<shared_ptr<Blob>> x, vector<shared_ptr<Blob>> y)
{
	shared_ptr<Layer> myLayer(NULL);
	layers_ = param.layers;   // 层名，param.layers类型为vector<string>
	ltypes_ = param.ltypes;    // 层类型 , param.ltypes类型为vector<string>

	X_train_ = x[0];
	Y_train_ = y[0];
	X_val_ = x[1];
	Y_val_ = y[1];

	for (int i = 0; i < layers_.size(); ++i)
	{
		data_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL); //为每一层创建前向计算要用到的3个Blob
		diff_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL); //为每一层创建前向计算要用到的3个Blob
		outShapes_[layers_[i]] = vector<int>(4);  //定义缓存，存储每一层的输出尺寸
	}

	vector<int> inShape = {
		param.batch_size,
		X_train_->get_channel(),
		X_train_->get_height(),
		X_train_->get_width(),
	};


	for (int i = 0; i < layers_.size()-1; i++)
	{
		string layer_name = layers_[i];
		string layer_type = ltypes_[i];
		if (layer_type == "Conv")
		{
			myLayer.reset(new ConvLayer);
		}
		else if (layer_type == "Relu")
		{
			myLayer.reset(new ReluLayer);
		}
		//else if (layer_type == "Tanh")
		//{

		//}
		else if (layer_type == "Pool")
		{
			myLayer.reset(new PoolLayer);
		}
		else if (layer_type == "Fc")
		{
			myLayer.reset(new FcLayer);
		}

		myLayers_[layer_name] = myLayer;
		myLayer->initLayer(inShape, layer_name, data_[layer_name], param.lparams[layer_name]);
		myLayer->calcShape(inShape, outShapes_[layer_name], param.lparams[layer_name]);
		inShape.assign(outShapes_[layer_name].begin(), outShapes_[layer_name].end());
		cout << layer_name << ".outShapes_->(" << outShapes_[layer_name][0] << "," << outShapes_[layer_name][1] << "," << outShapes_[layer_name][2] << "," << outShapes_[layer_name][3] << ")" << endl;

	}

	//data_["conv1"][1]->print("conv1W为：");
	//data_["conv1"][2]->print("conv1 b为：");
	//data_["fc1"][1]->print("fc1 W为：");
	//data_["fc1"][2]->print("fc1 b为：");
}

void Net::train(NetParam& net_param)
{

	int N = X_train_->get_batch_size();
	
	cout << "N = " << N << endl;
	int iter_per_epoch = N / net_param.batch_size;  //59000/200 = 295
	//总的批次数（迭代次数）= 单个epoch所含批次数 * epoch个数
	int num_batchs = iter_per_epoch * net_param.num_epochs;  // 295 * 2 = 590
	cout << "num_batchs(iterations) = " << num_batchs << endl;


	for (int iter = 0; iter < num_batchs; ++iter)
	//for (int iter = 0; iter < 1; ++iter)
	{
		//----------step1. 从整个训练集中获取一个mini-batch
		shared_ptr<Blob> X_batch;
		shared_ptr<Blob> Y_batch;

		X_batch.reset(new Blob(X_train_->subBlob((iter*net_param.batch_size) % N, ((iter+1)*net_param.batch_size) % N)));

		Y_batch.reset(new Blob(Y_train_->subBlob((iter*net_param.batch_size) % N, ((iter + 1)*net_param.batch_size) % N)));

		//----------step2. 用该mini-batch训练网络模型
		train_with_batch(X_batch, Y_batch, net_param);

		//----------step3. 评估模型当前准确率（训练集和验证集）
		evaluate_with_batch(net_param);
		printf("iter_%d    lr: %0.6f    loss: %f    train_acc: %0.2f%%    val_acc: %0.2f%%\n",
			iter, net_param.lr, loss_, train_accu_ * 100, val_accu_ * 100);
	}
}


void Net::train_with_batch(shared_ptr<Blob>& X, shared_ptr<Blob>& Y, NetParam& param, string mode)
{
	//------- step1. 将mini-batch填充到初始层的X当中
	data_[layers_[0]][0] = X;
	data_[layers_.back()][1] = Y;

	//------- step2. 逐层前向计算
	int n = layers_.size();  //层数
	for (int i = 0; i < n-1; ++i)
	{
		string layer_name = layers_[i];
		shared_ptr<Blob> out;
		myLayers_[layer_name]->forward(data_[layer_name], out, param.lparams[layer_name]);
		data_[layers_[i + 1]][0] = out; //当前层的输出等于下一层的输入

		//cout << "test " << endl;
	}

	//------- step3. softmax前向计算和计算代价值
	SoftmaxLossLayer::softmax_cross_entropy_with_logits(data_[layers_.back()], loss_, diff_[layers_.back()][0]);
	//cout << "loss_=" << loss_ << endl;   //第一次迭代后，损失值约为2.3
	if (mode == "TEST")//如果仅用于前向传播（做测试，不训练），则提前退出！不会再执行下面的反向传播和优化
		return;
	//------- step4. 逐层反向传播     //conv1<-relu1<-pool1<-fc1<-softmax
	for (int i = n - 2; i >= 0; --i)
	{
		string lname = layers_[i];
		myLayers_[lname]->backward(diff_[layers_[i + 1]][0], data_[lname], diff_[lname], param.lparams[lname]);
	}

	//----------step5. 参数更新（利用梯度下降）
	optimizer_with_batch(param);
}

void Net::optimizer_with_batch(NetParam& param)
{
	for (auto lname : layers_)    //for lname in layers_
	{

		//(1).跳过没有w和b的层
		if (!data_[lname][1] || !data_[lname][2])
		{
			continue;  //跳过本轮循环，重新执行循环（注意不是像break那样直接跳出循环）
		}

		//cout << "lname=" << lname << endl;
		//Sleep(1000);
		//(2).利用梯度下降更新有w和b的层
		for (int i = 1; i <= 2; ++i)
		{
			assert(param.optimizer == "sgd" || param.optimizer == "momentum" || param.optimizer == "rmsprop");//sgd/momentum/rmsprop
			//w:=w-param.lr*dw ;    b:=b-param.lr*db     ---->  "sgd"
			shared_ptr<Blob> dparam(new Blob(data_[lname][i]->size(), TZEROS));
			(*dparam) = -param.lr * (*diff_[lname][i]);
			(*data_[lname][i]) = (*data_[lname][i]) + (*dparam);
		}
	}
	//学习率更新
	if (param.lr_update)
		param.lr *= param.lr_decay;
}

void Net::evaluate_with_batch(NetParam& param)
{
	//(1).评估训练集准确率
	shared_ptr<Blob> X_train_subset;
	shared_ptr<Blob> Y_train_subset;
	int N = X_train_->get_batch_size();
	if (N > 1000)
	{
		X_train_subset.reset(new Blob(X_train_->subBlob(0, 1000)));
		Y_train_subset.reset(new Blob(Y_train_->subBlob(0, 1000)));
	}
	else
	{
		X_train_subset = X_train_;
		Y_train_subset = Y_train_;
	}
	train_with_batch(X_train_subset, Y_train_subset, param, "TEST");  //“TEST”，测试模式，只进行前向传播
	train_accu_ = calc_accuracy(*data_[layers_.back()][1], *data_[layers_.back()][0]);

	//(2).评估验证集准确率
	train_with_batch(X_val_, Y_val_, param, "TEST");  //“TEST”，测试模式，只进行前向传播
	val_accu_ = calc_accuracy(*data_[layers_.back()][1], *data_[layers_.back()][0]);
}

double Net::calc_accuracy(Blob& Y, Blob& Predict)
{
	//(1). 确保两个输入Blob尺寸一样
	vector<int> size_Y = Y.size();
	vector<int> size_P = Predict.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_Y[i] == size_P[i]);  //断言：两个输入Blob的尺寸（N,C,H,W）一样！
	}
	//(2). 遍历所有cube（样本），找出标签值Y和预测值Predict最大值所在位置进行比较，若一致，则正确个数+1
	int N = Y.get_batch_size();  //总样本数
	int right_cnt = 0;  //正确个数
	for (int n = 0; n < N; ++n)
	{
		//参考网址：http://arma.sourceforge.net/docs.html#index_min_and_index_max_member
		if (Y[n].index_max() == Predict[n].index_max())
			right_cnt++;
	}
	return (double)right_cnt / (double)N;   //计算准确率，返回（准确率=正确个数/总样本数）
}