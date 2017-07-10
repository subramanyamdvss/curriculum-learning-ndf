require 'cunn'
require 'cudnn'
require 'optim'
require 'nn'
require 'xlua'
require 'image'

---------------------DEFINING OPTIONS------------------
opt = lapp[[
   -s,--save                  (default "logs/")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)       learning rate
   --alpha                     (default 0.001)   learning rate for the ndf
   --gamma                     (default 0.95)      gamma used to find the cumulative reward
   --beta                     (default 0.001)      learning rate for actorcritic
   --episodes                  (default 500)       maximum number of episodes
   --maxiter                  (default 3000)      maximum number of iterations in an episode
   --tou                      (default 0.8)       accuracy threshold
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --epochs                   (3000)                no. of epochs
   --spiltval                 (default 50000)       spilt size to train ndf or start episode
    
   --backend                  (default cudnn)            backend
   --type                     (default cuda)          cuda/float/cl
   --startfrom                (default 0)             from which epoch should I start the training
   --eps                      (default 0.05)            epsilon for greedy policy 
   --eta                      (default 0.001)           eta during updating weights
   
   --loadprev                 (default 0)          load previous  epoch
]]


print(opt)
-------------------default tensor type-------------------------------------------------------------

torch.setdefaulttensortype('torch.CudaTensor')
--------------------classes------------------------------------------------------------------------
classes  =  {'0','1','2','3','4','5','6','7','8','9'}

--------------------LOADING THE DATASET------------------------------------------------------------
local mnist = require 'mnist'

local trainset = mnist.traindataset().data -- 60000x32x32
local testset = mnist.testdataset().data -- 10000x32x32
local targtrain = mnist.traindataset().label --60000
local targtest = mnist.testdataset().label   --10000 

-------------------defining networks of NDF policy and mnist model and criterions------------------
--for NDF the input has 24 features 10 label categories+ 10 log probabilities+ margin+ correct log probability+ normalized iteration+ average historical training accuracy.  


local ndf = nn.Sequential()
ndf:add(nn.Linear(24,12)):add(nn.Tanh()):add(nn.Linear(12,1)):add(nn.Sigmoid())
if filep(opt.save .. 'ndf.net') then
   ndf = torch.load(opt.save .. 'ndf.net')
end
   
local model = nn.Sequential()
ndf:add(nn.Linear(784,500)):add(nn.Tanh()):add(nn.Linear(500,10)):add(nn.LogSoftMax())
if filep(opt.save .. 'modelndf.net') then
   ndf = torch.load(opt.save .. 'modelndf.net')
end

local actorcritic = nn.Sequential()
local parallel = nn.ParallelTable()
parallel:add(nn.Identity())
parallel:add(nn.Linear(opt.batchSize,24,true))
actorcritic:add(parallel)
actorcritic:add(nn.MM())
actorcritic:add(nn.ReLU())
actorcritic:add(nn.Linear(opt.batchSize,1))
actorcritic:add(nn.Sigmoid())





local criterion = nn.ClassNLLCriterion()
local criteriondf = nn.BCECriterion()
-------------------------initializing state features and state-------------------------------------------------
local labelcateg = torch.Tensor(opt.batchSize,#classes):zero()
local logprobs = torch.Tensor(opt.batchSize,#classes):fill(0.1)
local correctprob = torch.Tensor(opt.batchSize,1):fill(0.1)
local margin = torch.Tensor(opt.batchSize,1):fill(0)
local normalizediter = torch.Tensor(opt.batchSize,1):fill(0)
local trainacc = torch.Tensor(opt.batchSize,1):fill(0)

function getstate()
   return torch.cat(labelcateg,logprobs,correctprob,margin,normalizediter,trainacc,2)
end
------------------------- weight initialization functions----------------------------------------------
function weightinit(model) 
   for k,v in pairs(model:findModules('nn.Linear')) do
      print({v})
      v.bias:fill(0)
      if k == 2 then
         v.bias:fill(2)
      end
      v.weight:normal(0,0.1)
   end
end
-------------------------confusion matrix and optimstate-------------------------------------------------------------

local confusion = optim.ConfusionMatrix(10)
local optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}
local filtbatch;
--------------------------feval function for model--------------------------------------------------------------

--------------------------feval function for ndf------------------------------------------------------

-------------------------testdev function-------------------------------------------------------------
function testdev(trainsetdev,targdev)
   local conf = optim.ConfusionMatrix(10)
   local outputs = model:forward(trainsetdev)
   conf:batchAdd(outputs,targdev)
   conf:updateValids()
   return conf.totalValid 
end

------------------------- train function--------------------------------------------------------------
local curreward = 0
local prevreward = 0
local filter = torch.Tensor(opt.batchSize):fill(1)
function trainpolicy() 

   --pick 50k from 60k and then start the episode 
   local trainsetd = torch.randperm(trainset:size(1)):long():split(50000)
   local trainsett = trainsetd[1]
   
   weightinit(ndf)
   for l = 1, opt.episodes do
      trainsetdevin = trainsett:long():split(5000)[1]
      local trainsetdev = trainset:index(1,trainsetdevin)
      local targdev = trainset:index(1,trainsetdevin)
      weightinit(model)
      local stop = false
      local T = 0
      while ~stop do
         T = T+1
         if T > opt.maxiter then
            stop = true
         end
         local shuffle = torch.randperm(trainsett:size(1)):long()
         local batchindx = trainsett:index(1,shuffle):split(opt.batchSize)[1]
         collectgarbage()
         local batch = trainset:index(1,batchindx)
         batch = torch.reshape(batch,batch:size(1),batch:size(2)*batch:size(3))
         local targets = targtrain:index(1,batchindx)
         --forward pass every instance in the batch and get state of that instance 
         local states;
         local prevstates = getstate()
         local prevfilter = filter
         for i = 1,batch:size(1) do
            local output = model:forward(batch[i])
            
            logprobs[i] = torch.log(output)
            labelcateg[i]:zero()
            labelcateg[i][targets[i]] = 1
            correctprob[i][1] = logprobs[i][targets[i]]
            val,indx = torch.max(output)
            margin[i][1] = output[targets[i]]-val
            normalizediter[i][1] = T/opt.maxiter
            trainacc[i][1] = confusion.totalValid
            states = getstate()
         end
         --find out filtered batch
         ndfprobs = ndf:forward(states)
         filter = torch.ge(ndfprobs,0.5)
         local num = torch.sum(filter)
         local indx = torch.CudaLongTensor(num)
         local j = 0;
         for i = 1,opt.batchSize do
            if filter[i] == 1 then
               j = j + 1
               indx[j] = i
            end
         end

         batch = batch:index(1,indx)
         targets = targets:index(1,indx)
         collectgarbage()
         if filtbatch then
            filtbatch = torch.cat(filtbatch,batch,1)
         else
            filtbatch = batch
         end
         --if filtered batch has more than M instances then train that network with that batch
         if filtbatch:size(1)>=opt.batchSize then
            filtbatch = filtbatch:split(opt.batchSize)
            batch = filtbatch[1]
            filtbatch = filtbatch[2]
            local parametersm,gradParametersm = model:getParameters()
            local feval = function(x)
               if x ~= parametersm then parametersm:copy(x) end
               gradParametersm:zero()
               local outputs = model:forward(batch)
               f = criterion:forward(outputs, targets)
               local df_do = criterion:backward(outputs, targets)
               model:backward(inputs, df_do)
               confusion:batchAdd(outputs, targets)
               return f,gradParametersm
            end
            optim.adam(feval,parametersm)
            --find the reward 
            local tmp = torch.randperm(trainsetdev:size(1)):split(1000)[1]
            curreward =  testdev(trainsetdev:index(1,tmp),targdev:index(1,tmp))
            
            
            --update the ndf and actor critic
            --first you need to find Q
            local Q = actorcritic:forward{states,filter}
            local parametersn,gradParametersn = ndf:getParameters()
            local ndfeval = function(x)
               if x ~= parametersn then parametersn:copy(x) end
               gradParametersn:zero()
               local outputs = ndf:forward(states)
               f = criteriondf:forward(outputs, torch.Tensor(opt.batchSize):fill(1))
               local df_do = criteriondf:backward(outputs, torch.Tensor(opt.batchSize):fill(1))
               ndf:backward(states, Q*df_do)
               return f,gradParametersn
            end
            optim.adam(ndfeval,parametersn)
            --so you found the Q now substitute it in the function and multiply it with the gradients
            --after updating the ndf you need to update actor critic.
            local q = prevreward + opt.gamma*actorcritic:forward(states,filter)-actorcritic:forward(prevstates,prevfilter)
            local parametersa,gradParametersa = actorcritic:getParameters()
            local actoreval = function(x)
               if x ~= parametersa then parametersa:copy(x) end
               gradParametersa:zero()
               local outputs = ndf:forward{prevstates,prevfilter}
               actorcritic:backward({prevstates,prevfilter}, q)
               return f,gradParametersa
            end
            optim.adam(actoreval,parametersa)
         end
         
         prevreward = curreward

      end
   end
end

function train()
   --you should for each iteration update the model, but using the policy
   --instead of using epochs style why not random picking of the batch.
   --pick a batch and filter it if < 128 go ahead to another batch.
   local filtbatch;
   local stop = false
   while ~stop do
      --find out the state
      local rnd = torch.randperm(trainset:size(1)):long():split(opt.batchSize)[1]
      local batch = trainset:index(1,rnd)
      local targets = targtrain:index(1,rnd)
      local states;  
      for i = 1,batch:size(1) do
         local output = model:forward(batch[i])
         logprobs[i] = torch.log(output)
         labelcateg[i]:zero()
         labelcateg[i][targets[i]] = 1
         correctprob[i][1] = logprobs[i][targets[i]]
         val,indx = torch.max(output)
         margin[i][1] = output[targets[i]]-val
         normalizediter[i][1] = T/opt.maxiter
         trainacc[i][1] = confusion.totalValid
         states = getstate()
      end
         --find out filtered batch
      ndfprobs = ndf:forward(states)
      filter = torch.ge(ndfprobs,0.5)
      local num = torch.sum(filter)
      local indx = torch.CudaLongTensor(num)
      local j = 0;
      for i = 1,opt.batchSize do
         if filter[i] == 1 then
            j = j + 1
            indx[j] = i
         end
      end

      batch = batch:index(1,indx)
      targets = targets:index(1,indx)
      collectgarbage()
      if filtbatch then
         filtbatch = torch.cat(filtbatch,batch,1)
      else
         filtbatch = batch
      end
      --if filtered batch has more than M instances then train that network with that batch
      if filtbatch:size(1)>=opt.batchSize then
         filtbatch = filtbatch:split(opt.batchSize)
         batch = filtbatch[1]
         filtbatch = filtbatch[2]
         local parametersm,gradParametersm = model:getParameters()
         local feval = function(x)
            if x ~= parametersm then parametersm:copy(x) end
            gradParametersm:zero()
            local outputs = model:forward(batch)
            f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            confusion:batchAdd(outputs, targets)
            return f,gradParametersm
         end
         optim.adam(feval,parametersm)
      end
   end
end

