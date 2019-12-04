# mnist-pytorch

PyTorch implementation of a begginer classification task on [MNIST](http://yann.lecun.com/exdb/mnist/) and [FASHION-MNSIT](https://github.com/zalandoresearch/fashion-mnist) dataset with a 3 Layers Neural Network model.  
I tested this experiments for getting used to PyTorch.  

## Dependencies
- Ubuntu 18.04 LTS
- PyTorch 1.0.1
- NumPy 1.16.3

## Usage
Execute the scripts for training on MNIST,
```
$ python mnist.py 100 0.1
```

Execute the scripts for training on FASHION-MNIST,
```
$ python mnist.py 100 0.1 fashion
```

## Results
### MNIST
- train loss

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/324488/81b0eb15-7c82-5aa0-5521-80eebb4fb039.png" width=50%>

- test loss

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/324488/a136f87b-3361-89ee-38f1-d7126e39ff92.png" width=50%>

- test accuracy

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/324488/788dd92d-3b98-95c7-2fe1-280f88f47e97.png" width=50%>


### FASHION-MNIST
- train loss

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/324488/4fcdfd06-b846-1130-f76f-d7a102a54396.png" width=50%>

- test loss

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/324488/0cb7bfbd-5016-8e0b-a551-b33a0c6caadf.png" width=50%>

- test accuracy

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/324488/a03c9685-b345-e808-72e2-d73ad81597f4.png" width=50%>
