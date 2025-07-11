# 算子功耗测量使用方法
## 一、环境安装
1. 安装python
2. 安装pytorch、torchvision等支持各类算子所需的环境
3. 确保正确安装NVIDIA驱动，或者昆仑芯GPU的驱动，最终能使用命令nvidia-smi或者kml-smi
4. 确保正确安装turbostat，Linux系统上的CPU监测工具，最终能使用命令sudo turbostat --interval 1即可
5. 尝试运行命令python main.py，如无报错即说明安装成功

## 二、基础算子测试
### 方法一
1. 使用命令python main.py <算子名(模型名),...> <测试次数>  

例如python main.py softmax,alex_net,conv 50，表示测试算子softmax，模型alex_net和算子conv各50次
其中所有算子必须是本项目提供的基础算子或者是用户自己实现并注册的算子！  
实现和注册算子的方法详见使用手册！
### 方法二
1. 直接使用命令python main.py  
2. 修改main.py中主函数中变量op_names和num_samples的值  

其中，op_names表示要测试的算子和模型的名称，默认是所有算子和模型，是一个字符串列表  
而num_samples表示op_names中每个算子要测试的次数
## 三、查看结果
最终的结果将以为文件名"算子名.csv"保存在与src同级的results文件夹下 
同时，程序运行时将会产生一些中间文件，保存在与src同级的temp文件夹下，不必理会，程序运行完后即可清理  
如果使用了models中的，可能会下载一些数据集，保存在与src同级的data文件夹下，请勿删除，否则运行程序时将重新下载
## 四、基础算子及扩展算子
项目提供了21种常见的算子和2种常用的模型，其中21种算子其中包括：avg_pooling,conv,elu,linear_layer,  
max_pooling,relu,silu,leaky_relu,spmm,flatten,cat,lay_norm,embedding,  
positional_encoding,roi_align,nms,add,softmax,lstm  
2种模型包括：alex_net,vgg

它们的含义详见说明文档  
如果用户需要测试其他的算子和模型，用户也可以自定义算子和模型，自定义算子和模型的方法详见使用手册！
