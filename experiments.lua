require 'nn'
require 'optimizer'
require 'train'
require 'logging'
require 'data'


Experiment = {
    useCuda = false,
    numGPUs = 1,
    directories = {train="train", validation="validation", test="test"},
    datasets = {},
    data_root = "/home/ubuntu/ebs_disk",
    validationSize = 2000,
    validation_interval = 2000,
    batchSize = 32
}

function Experiment:new(dict)
    local dict = dict or {}
    local result = {}
    for key, value in pairs(self) do
        result[key] = value
    end
    for key, value in pairs(dict) do
        result[key] = value
    end
    self.__index = self
    setmetatable(result, self)
    return result 
end

basicGoExperiment = Experiment:new { 
    name = "basicGoExperiment",

    kernels = {5},
    strides = {1},
    channels = {37},
    numLayers = 3,
    channelSize = 64,
    
    rate = .01,
    rateDecay = 1e-7,

    criterion = nn.ClassNLLCriterion()
}

function Experiment:init_data()
    for group, directory in pairs(self.directories) do
        self.datasets[group] = Dataset:new{root=self.data_root, directory=directory}
        self.datasets[group]:init()
    end
end

dont_serialize = {datasets}

function Experiment:pickle()
    local pickled = {}
    for key, value  in pairs(self) do
        if dont_serialize[key] ~= 1 then pickled[key] = value end
    end
    return pickled
end

function Experiment:unpickle(pickled)
    result = self:new()
    for key, value in pairs(pickled) do
        if dont_serialize[key] ~=1 then result[key] = value end
    end
    result:init_data()
    return result
end


function Experiment:init()
    self.id = torch.uniform()
    self.iterations = 0
    self.initialized = true
    self.optimizer = SGD.new(self.rate, self.rateDecay)
    self.validation_costs = {}

    self:init_data()

    print("initializing model...")
    -- set up model
    -- eventually, this should only be done if the experiment is of the appropriate type
    -- different types of experiments may have different setups
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
        self:init()
    end

    local start_time = sys.clock()
    local train_cost, validation_cost = train(self, params)
    local runningTime = sys.clock() - start_time

    log(self, train_cost, runningTime)
end 

function Experiment:save(opt_filename)
    local filename = opt_filename or self.id .. '.model'
    torch.save(filename, self:pickle())
end

function Experiment:load(filename)
    return self:unpickle(torch.load(filename))
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


