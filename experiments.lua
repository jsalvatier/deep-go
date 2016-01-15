require 'nn'
require 'optimizer'
require 'train'
require 'logging'
require 'godata'

Experiment = {useCuda = true, numGPUs = 1, dataset = GoDataset, group = 'train'}
function Experiment:new(dict)
    local dict = dict or {}
    self.__index = self
    setmetatable(dict, self)
    
    return dict
end

basicGoExperiment = Experiment:new {}

function basicGoExperiment:init()
    self.id = torch.uniform()
    self.iterations = 0
    self.initialized = true
    self.optimizer = SGD.new(self.rate, self.rateDecay)
    self.validation_costs = {}

    -- set up model
    for i = 2, self.numLayers do
        table.insert(self.kernels, 3)
        table.insert(self.strides, 1)
        table.insert(self.channels, self.channelSize)
    end
    table.insert(self.channels, 1)
    self.model = getBasicModel(self.numLayers, self.kernels, self.channels)

    -- send model to GPU, and parallelize across GPUs
    if self.useCuda then
	require 'cutorch'
	require 'cunn'
        self.model = self.model:cuda()
        self.criterion = self.criterion:cuda()
        self.model = makeDataParallel(self.model, self.numGPUs)
    end

    -- this has to be called only once, and after the model has been
    -- moved to the GPU
    self.modelParameters, self.grads = self.model:getParameters()
end

function Experiment:run(params)
    assert(params.iters > 0)

    if not self.initialized then
        print("initializing model...")
        self:init()
    end

    local start_time = sys.clock()
    local train_cost, validation_cost = train(experiment, params)
    local runningTime = sys.clock() - start_time

    log(self, train_cost, runningTime, validation_cost)
end 

function Experiment:save(opt_filename)
    local filename = opt_filename or self.id .. '.model'

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

function makeDataParallel(model, nGPU)
   if nGPU > 1 then
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      local oldGPU = cutorch.getDevice()
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
      cutorch.setDevice(oldGPU)
   end
   return model
end


