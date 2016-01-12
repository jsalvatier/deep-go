require"data"
require"model"
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'


nbatches = 2 
batchsize = 32

--data
dataset = GoDataset

--model
model = basic

--criterion

criterion = nn.ClassNLLCriterion()

optimState = {
      learningRate = .003,
      weightDecay = 0,
      momentum = 0,
      learningRateDecay = 1e-7
   }
optimMethod = optim.sgd

--train
--[[
parameters, gradParameters = model:getParameters()

for batches = 1, nbatches do 
    batch = dataset:minibatch("train", batchsize)

    function feval(x) 
        if x ~= parameters then
            parameters:copy(x)
        end
        gradParameters:zero()

        output = model:forward(batch.input)
        f = criterion:forward(output, batch.output)
        
        -- estimate df/dW
        df_do = criterion:backward(output, batch.output)
        grad = model:backward(batch.input, df_do)
        
        gradParameters:div(batch.input:size()[1])
        f = f/batch.input:size()[1]
        return f, gradParameters
    end 
    optimMethod(feval, parameters, optimState)
end
--]]
--evaluate
