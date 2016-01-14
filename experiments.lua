require 'nn'
require 'optimizer'
require 'train'
require 'logging'
require 'godata'

Experiment = {useCuda = true, numGPUs = 1, dataset = GoDataset, group = 'train'}
function Experiment:new(dict)
    dict = dict or {}
    self.__index = self
    setmetatable(dict, self)
    
    return dict
end

basicGoExperiment = Experiment:new {}

function basicGoExperiment:init()
    for i = 2, self.numLayers do
        table.insert(self.kernels, 3)
        table.insert(self.strides, 1)
        table.insert(self.channels, self.channelSize)
    end
    table.insert(self.channels, 1)

    self.optimizer = SGD.new(self.rate, self.rateDecay)
    self.model = getBasicModel(self.numLayers, self.kernels, self.channels)
    self.modelParameters, self.grads = self.model:getParameters()

    self.iterations = 0
    self.initialized = true
end

function Experiment:run(params)
    -- if GPU usage is unspecified, inherit default from experiment
    params.numGPUs = params.numGPUs or self.numGPUs
    params.useCuda = params.useCuda or self.useCuda

    assert(params.iters > 0)

    if not self.initialized then
        print("initializing model...")
        self:init()
    end

    local start_time = sys.clock()
    local train_cost, validation_cost = train(experiment, params)
    local runningTime = sys.clock() - start_time

    self.iterations = self.iterations + params.iters
    log(self, train_cost, runningTime, validation_cost)
end 

function Experiment:save(filename)
    torch.save(filename, self)
end

function getBasicModel(numLayers, kernels, channels) 
    local smodel = nn.Sequential()
    for layer = 1, numLayers do
        local padding = (kernels[layer] - 1)/2
        smodel:add(nn.SpatialZeroPadding(padding, padding, padding, padding))
        smodel:add(nn.SpatialConvolutionMM(channels[layer], channels[layer+1], kernels[layer], kernels[layer]))

        local d1 = channels[layer+1]
        local d2 = 19
        local d3 = 19
        smodel:add(nn.Reshape(d1*d2*d3))
        smodel:add(nn.Add(d1*d2*d3))
        smodel:add(nn.Reshape(d1, d2, d3))

        smodel:add(nn.ReLU())
    end
    
    smodel:add(nn.Reshape(19*19))
    smodel:add(nn.LogSoftMax())
    return smodel
end



