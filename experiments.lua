require 'nn'
require 'model'
require 'optimizer'
require 'train'
require 'logging'

Experiment = {}
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

    self.iterations = 0
end

function Experiment:run(iterations)
    print(iterations) 
    
    start_time = sys.clock()
    train_cost,validation_cost = train(self.model, self.criterion, 
        self.batchSize, self.iterations, self.optimizer, 
        self.useCuda, self.dataset, self.group)
    runningTime = sys.clock() - start_time

    self.iterations = self.iterations + iterations
    log(self, train_cost, runningTime, validation_cost)
end 

function Experiment:save(filename)
    torch.save(filename, self)
end

function Experiment:load(filename)
    self = torch.load(filename)
end

function getBasicModel(numLayers, kernels, channels) 
    smodel = nn.Sequential()
    for layer = 1, numLayers do
        local padding = (kernels[layer] - 1)/2
        smodel:add(nn.SpatialZeroPadding(padding, padding, padding, padding))
        smodel:add(nn.SpatialConvolutionMM(channels[layer], channels[layer+1], kernels[layer], kernels[layer]))

        d1 = channels[layer+1]
        d2 = 19
        d3 = 19
        smodel:add(nn.Reshape(d1*d2*d3))
        smodel:add(nn.Add(d1*d2*d3))
        smodel:add(nn.Reshape(d1, d2, d3))

        smodel:add(nn.ReLU())
    end
    
    smodel:add(nn.Reshape(19*19))
    smodel:add(nn.LogSoftMax())
    return smodel
end



