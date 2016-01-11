require 'nn'
require 'math'
require 'debug'

conv = 0
pool = 1
sizes = {{32, 32}}
kernels = {5, 3, 7, 3}
strides = {1, 2, 1, 2}
channels = {3, 16, 16, 64, 64}
types = {conv, pool, conv, pool}

function shrink(dims, k)
    result = {}
    for i, d in ipairs(dims) do
        result[i] = dims[i] - k + 1
    end
    return result
end

function downsampled_size(dims, k, d)
    result = {}
    for i, dim in ipairs(dims) do
        result[i] = math.floor( (dim - k) / d + 1 )
    end
    return result
end

function sizeOf(n)
    return sizes[n][1] * sizes[n][2] * channels[n]
end

model = nn.Sequential()

numlayers = 3

for layer = 1, numlayers-1 do
    if types[layer] == conv then
        model:add(nn.SpatialConvolutionMM(channels[layer], channels[layer+1], kernels[layer], kernels[layer]))
        sizes[layer+1] = shrink(sizes[layer], kernels[layer])
        model:add(nn.ReLU())
    elseif types[layer] == pool then
        model:add(nn.SpatialLPPooling(channels[layer], 2, kernels[layer], kernels[layer], strides[layer], strides[layer]))
        sizes[layer+1] = downsampled_size(sizes[layer], kernels[layer], strides[layer])
    end
end

model:add(nn.Reshape(sizeOf(numlayers)))
model:add(nn.Linear(sizeOf(numlayers), 20))
model:add(nn.ReLU())
model:add(nn.Linear(20, 10))

model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

AdagradOptimizer = {}

function AdagradOptimizer:reset()
    self.grad_ms = grads:clone():fill(1)
    self.decay = 0.95
    self.rate = 0.001
end

function AdagradOptimizer:step(grad)
    self.grad_ms:mul(self.decay)
    self.grad_ms = self.grad_ms:add(1 - self.decay, torch.cmul(grad,grad))
    parameters:add(-self.rate, torch.cdiv(grad, torch.sqrt(self.grad_ms)))
end

SGD = {}
function SGD:reset()
    self.rate = 0.003
    self.rate_decay = 1e-7
end

function SGD:step()
    parameters:add(-self.rate, grad)
    self.rate = self.rate * (1 - self.rate_decay)
end
SGD:reset()

function useModel(newModel)
    model = newModel
    parameters, grads = model:getParameters()
    AdagradOptimizer:reset()
end

useModel(model)

function makeBatch(dataset, batchsize)
    inputs = {}
    outputs = {}
    for i = 1, batchsize do
        k = math.random(1, dataset:size())
        inputs[i] = dataset.data[{{k,k}, {}, {}, {}}]:double()
        outputs[i] = dataset.labels[k]
    end
    return torch.cat(inputs, 1):double(),  torch.Tensor(outputs):double()
end

function evalBatch(dataset, batchsize)
    inputs, targets = makeBatch(dataset, batchsize)
    return eval(inputs, targets)
end

function eval(inputs, targets)
    grads:zero()
    outputs = model:forward(inputs:double())
    cost = criterion:forward(outputs, targets)
    df_do = criterion:backward(outputs, targets)
    model:backward(inputs, df_do)
    return cost, grads
end

function validate(data)
    local cost, grads = eval(data.data:double(), data.labels:double())
    return cost
end

function train(dataset, batchsize, iters, optimizer)
    optimizer = optimizer or AdagradOptimizer
    for i = 1, iters do
        cost, grad = evalBatch(dataset, batchsize)
        optimizer:step(grad)
    end
end

function accuracy(dataset)
    probs = model:forward(dataset.data:double())
    cost = criterion:forward(probs, dataset.labels)
    maxs, preds = torch.max(probs, 2)
    local errors = 0
    for i = 1, dataset:size() do
        if preds[i][1] ~= dataset.labels[i] then
            errors = errors + 1
        end
    end
    return 1 - (errors * 1.0 / dataset:size()), cost
end
