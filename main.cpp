#include "Net.hpp"
#include "Blob.hpp"
#include "Utils.hpp"
#include "Net.hpp"
#include <iostream>
#include <string>
using namespace std;

void trainModel(string configFile, shared_ptr<Blob> images, shared_ptr<Blob> labels)
{

	NetParam net_param;

	//1.读取myModel.json到内存中
	net_param.readNetParam(configFile);


	//2.打印我们的这些参数
	cout << "learning rate =  " << net_param.lr << endl;
	cout << "batch size =  " << net_param.batch_size << endl;
	vector <string> layers_ = net_param.layers;
	vector <string> ltypes_ = net_param.ltypes;

	for (int i = 0; i < layers_.size(); ++i)
	{
		cout << "layer = " << layers_[i] << " ; " << "ltype = " << ltypes_[i] << endl;
	}

	//1. 将60000张图片以59:1的比例划分为训练集（59000张）和验证集（1000张）
	shared_ptr<Blob> X_train(new Blob(images->subBlob(0, 59000)));  //左闭右开区间，即[ 0, 59000 )
	shared_ptr<Blob> Y_train(new Blob(labels->subBlob(0, 59000)));
	shared_ptr<Blob> X_val(new Blob(images->subBlob(59000, 60000)));
	shared_ptr<Blob> Y_val(new Blob(labels->subBlob(59000, 60000)));

	vector<shared_ptr<Blob>> XX{ X_train, X_val };
	vector<shared_ptr<Blob>> YY{ Y_train, Y_val };

	Net net;
	net.init(net_param, XX, YY);

	net.train(net_param);
}
int main(int argc, char** argv){
	//cout << "test " << endl;
	//exit(1);
	//Blob blob(1, 2, 3, 4, TRANDN);
	//blob.print("Blob 里面的数据是：\n");
	
	shared_ptr<Blob> images(new Blob(60000, 1, 28, 28, TZEROS));
	shared_ptr<Blob> labels(new Blob(60000, 10, 1, 1, TZEROS));
	ReadMnistData("mnist_data/train/train-images.idx3-ubyte", images);  //读取data
	ReadMnistLabel("mnist_data/train/train-labels.idx1-ubyte", labels);   //读取label

	//vector<cube>& list0 = images->get_data();
	//vector<cube>& list1 = labels->get_data();
	//for (int i = 0; i < 2; ++i)
	//{
	//	list0[i].print("images：\n");
	//	list1[i].print("labels：\n");
	//}

	string configFile = "./myModel.json";

	trainModel(configFile, images, labels);

	system("pause");
}