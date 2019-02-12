#ifndef __NET_HPP__
#define __NET_HPP__

#include "Layer.hpp"
#include "Blob.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

using std::unordered_map;
using std::string;
using std::vector;
using std::shared_ptr;

struct NetParam      //c++中，struct跟class用法基本一致！主要区别是继承后的数据访问权限。
{
	/*学习率*/
	double lr;
	/*学习率衰减系数*/
	double lr_decay;
	/*优化算法,:sgd/momentum/rmsprop*/
	std::string optimizer;
	/*momentum系数 */
	double momentum;
	/*epoch次数 */
	int num_epochs;
	/*是否使用mini-batch梯度下降*/
	bool use_batch;
	/*每批次样本个数*/
	int batch_size;
	/*每隔几个迭代周期评估一次准确率？ */
	int eval_interval;
	/*是否更新学习率？  true/false*/
	bool lr_update;
	/* 是否保存模型快照；快照保存间隔*/
	bool snap_shot;
	/*每隔几个迭代周期保存一次快照？*/
	int snapshot_interval;
	/* 是否采用fine-tune方式训练*/
	bool fine_tune;
	/*预训练模型文件.gordonmodel所在路径*/
	string preTrainModel;

	/*层名*/
	vector <string> layers;
	/*层类型*/
	vector <string> ltypes;

	/*无序关联容器, 保存层信息*/
	unordered_map<string, Param> lparams;


	void readNetParam(string file);


};


class Net
{
public:
	void init(NetParam& param, vector<shared_ptr<Blob>> x, vector<shared_ptr<Blob>> y);
	void train(NetParam& net_param);
	void train_with_batch(shared_ptr<Blob>& X, shared_ptr<Blob>& Y, NetParam& param, string mode = "TRAIN");

	void optimizer_with_batch(NetParam& param);
	void evaluate_with_batch(NetParam& param);
	double calc_accuracy(Blob& Y, Blob& Predict);

private:
	// 训练集
	shared_ptr<Blob> X_train_;
	shared_ptr<Blob> Y_train_;
	// 验证集
	shared_ptr<Blob> X_val_;
	shared_ptr<Blob> Y_val_;

	vector<string> layers_; //层名
	vector<string> ltypes_; //层类型
	double loss_;
	double train_accu_;
	double val_accu_;

	unordered_map<string, shared_ptr<Layer>> myLayers_;

	unordered_map<string, vector<shared_ptr<Blob>>> data_; //前向计算需要用到的Blob data_[0]=X,  data_[1]=W,data_[2] = b;
	unordered_map<string, vector<shared_ptr<Blob>>> diff_; //前向计算需要用到的Blob data_[0]=X,  data_[1]=W,data_[2] = b;
	unordered_map<string, vector<int>> outShapes_; //存储每一层的输出尺寸
};

#endif