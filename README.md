# PredNet
This is a implimentation of [PredNet][] on chainer.

[PredNet]: http://arxiv.org/abs/1605.08104 "Lotter, William, Gabriel Kreiman, and David Cox. \"Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning.\" arXiv preprint arXiv:1605.08104 (2016)."

#Testd on
*Ubuntu 14.04  
*Python 2.7.6  
*chainer 1.9.1  
*CUDA Toolkit 7.5  
*cuDNN v5  

#Demo
1. git clone  
$ git clone https://github.com/quadjr/PredNet.git  
$ cd PredNet  

2. Prepare Dataset  
This command will download The KITTI Dataset(about 47GB), unzip, and make image lists.  
$ ./scripts/prepare_kitti.sh  
You can intterupt and resume downloading.  

3. Train  
On a CPU  
$ python main.py -i dataset/train_list.txt  
On a GPU  
$ python main.py -i dataset/train_list.txt -g 0  
  
Model files and some images will be generated in models/ and images/ directory.   
Image suffix means x (input), y (predicted), z (correct).  

4. Test  
$ python main.py -i dataset/test_list.txt --test --initmodel models/???.model  
Plase specify the model file with option --initialmodel.  
Test result will be generated in images/ directory.  
Image suffix means x (input), y (predicted).  

#TODO
*Confirm the learning strategy.  

