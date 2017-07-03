require 'cunn'
require 'cudnn'
require 'optim'
require 'nn'
require 'xlua'
require 'image'

---------------------DEFINING OPTIONS------------------
opt = lapp[[
   -s,--save                  (default "logs3/vgg")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)       learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default cudnn)            backend
   --type                     (default cuda)          cuda/float/cl
   --startfrom                (default 0)             from which epoch should I start the training
   --eps                      (default 0.05)            epsilon for greedy policy 
   --eta                      (default 0.001)           eta during updating weights
   --beta                     (default 0)               beta while calculating reward bar
   --loadprev                 (default 0)          load previous  epoch
]]


print(opt)


--------------------LOADING THE DATASET-----------
local mnist = require 'mnist'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

print({trainset}) -- to retrieve the size
print({testset}) -- to retrieve the size
