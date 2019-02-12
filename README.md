# pytorch-peleenet
Simple Code Implementation of ["PeleeNet" in NeurIPS 2018](https://papers.nips.cc/paper/7466-pelee-a-real-time-object-detection-system-on-mobile-devices.pdf) architecture using PyTorch.

<p align="center">
  <img width="700" src="https://github.com/hoya012/pytorch-peleenet/blob/master/assets/architecture.png" "PeleeNet architecture">
</p>

For simplicity, i write codes in `ipynb`. So, you can easliy test my code.

I tested PeleeNet using not ImageNet but **CIFAR-10** because of very very long training time & GPU resourece..

*Last update : 2019/2/12*

## Contributor
* hoya012

## 0. Requirements
```
python=3.5
numpy
matplotlib
torch=1.0.0
torchvision
torchsummary
```

## 1. Usage
You only run `PeleeNet-PyTorch.ipynb`. 

Or you can use Google Colab for free!! This is [colab link](https://colab.research.google).

After download ipynb, upload to your google drive. and run!

For training, testing, i used `CIFAR-10` Dataset.

## 2. Paper Review & Code implementation Blog Posting (Korean Only)
[“Pelee Tutorial [1] Paper Review & Implementation details”](https://hoya012.github.io/blog/Pelee-Tutorial-1/)  

## 3. PeleeNet and other blocks impelemtation.
In PeleeNet, there are some changes compared to DenseNet. This is my simple implemenatation.

### Dense layer (Bottleneck layer in DenseNet)
In PeleeNet, we will use two-way dense layer. 

```
class dense_layer(nn.Module):
  def __init__(self, nin, growth_rate, drop_rate=0.2):    
      super(dense_layer, self).__init__()
      
      self.dense_left_way = nn.Sequential()
      
      self.dense_left_way.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=growth_rate*2, kernel_size=1, stride=1, padding=0, bias=False))
      self.dense_left_way.add_module('conv_3x3', conv_bn_relu(nin=growth_rate*2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
            
      self.dense_right_way = nn.Sequential()
      
      self.dense_right_way.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=growth_rate*2, kernel_size=1, stride=1, padding=0, bias=False))
      self.dense_right_way.add_module('conv_3x3_1', conv_bn_relu(nin=growth_rate*2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
      self.dense_right_way.add_module('conv_3x3 2', conv_bn_relu(nin=growth_rate//2, nout=growth_rate//2, kernel_size=3, stride=1, padding=1, bias=False))
      
      self.drop_rate = drop_rate
      
  def forward(self, x):
      left_output = self.dense_left_way(x)
      right_output = self.dense_right_way(x)

      if self.drop_rate > 0:
          left_output = F.dropout(left_output, p=self.drop_rate, training=self.training)
          right_output = F.dropout(right_output, p=self.drop_rate, training=self.training)
          
      dense_layer_output = torch.cat((x, left_output, right_output), 1)
            
      return dense_layer_output
```

### Transition layer
The only difference is `bn_relu_conv` to `conv_bn_relu` compared to [DenseNet Implementation](https://github.com/hoya012/pytorch-densenet).

```
class Transition_layer(nn.Sequential):
  def __init__(self, nin, theta=1):    
      super(Transition_layer, self).__init__()
      
      self.add_module('conv_1x1', conv_bn_relu(nin=nin, nout=int(nin*theta), kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
```

### StemBlock
![](https://github.com/hoya012/pytorch-peleenet/blob/master/assets/stem_block.png)

PeleeNet use Stem Block. This is my implementation of Stem Block.

```
class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        
        self.conv_3x3_first = conv_bn_relu(nin=3, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.conv_1x1_left = conv_bn_relu(nin=32, nout=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3x3_left = conv_bn_relu(nin=16, nout=32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.max_pool_right = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv_1x1_last = conv_bn_relu(nin=64, nout=32, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out_first = self.conv_3x3_first(x)
        
        out_left = self.conv_1x1_left(out_first)
        out_left = self.conv_3x3_left(out_left)
        
        out_right = self.max_pool_right(out_first)
        
        out_middle = torch.cat((out_left, out_right), 1)
        
        out_last = self.conv_1x1_last(out_middle)
                
        return out_last
```

### DenseBlock
```python
class DenseBlock(nn.Sequential):
  def __init__(self, nin, num_dense_layers, growth_rate, drop_rate=0.0):
      super(DenseBlock, self).__init__()
                        
      for i in range(num_dense_layers):
          nin_dense_layer = nin + growth_rate * i
          self.add_module('dense_layer_%d' % i, dense_layer(nin=nin_dense_layer, growth_rate=growth_rate, drop_rate=drop_rate))
```

### PeleeNet
This is class implementation of PeleeNet. I use default setting of hyper-parameters.
**The only difference is num_classes. (ImageNet: 1000 vs CIFAR-10: 10)**

```python
class PeleeNet(nn.Module):
    def __init__(self, growth_rate=32, num_dense_layers=[3,4,8,6], theta=1, drop_rate=0.0, num_classes=10):
        super(PeleeNet, self).__init__()
        
        assert len(num_dense_layers) == 4
        
        self.features = nn.Sequential()
        self.features.add_module('StemBlock', StemBlock())
        
        nin_transition_layer = 32
        
        for i in range(len(num_dense_layers)):
            self.features.add_module('DenseBlock_%d' % (i+1), DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[i], growth_rate=growth_rate, drop_rate=0.0))
            nin_transition_layer +=  num_dense_layers[i] * growth_rate
            
            if i == len(num_dense_layers) - 1:
                self.features.add_module('Transition_layer_%d' % (i+1), conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer*theta), kernel_size=1, stride=1, padding=0, bias=False))
            else:
                self.features.add_module('Transition_layer_%d' % (i+1), Transition_layer(nin=nin_transition_layer, theta=1))
        
        self.linear = nn.Linear(nin_transition_layer, num_classes)
        
    def forward(self, x):
        stage_output = self.features(x)
        
        global_avg_pool_output = F.adaptive_avg_pool2d(stage_output, (1, 1))  
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)
                
        output = self.linear(global_avg_pool_output_flat)
        
        return output
```

## 4. Training phase
PeleeNet use Cosine Annealing Learning Rate Schedueling.

This is simple implemenation using torch.optim.lr_scheduler.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)
learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
running_loss = 0.0

  
for epoch in range(num_epoch):  
    learning_rate_scheduler.step()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
                
        show_period = 100
        if i % show_period == 0:    # print every "show_period" mini-batches
            print('[%d, %5d/50000] loss: %.7f, lr: %.7f' %
                  (epoch + 1, (i + 1)*batch_size, running_loss / show_period, learning_rate_scheduler.get_lr()[0]))
            running_loss = 0.0
            
    # validation part
    correct = 0
    total = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('[%d epoch] Accuracy of the network on the validation images: %d %%' % 
          (epoch + 1, 100 * correct / total)
         )

print('Finished Training')
```
