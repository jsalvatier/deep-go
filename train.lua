require 'godata'

function eval(model, inputs, targets, criterion)
    grads:zero()
    preds = model:forward(inputs)

    cost = criterion:forward(preds, targets)
    df_do = criterion:backward(preds, targets)
    model:backward(inputs, df_do)
    return cost, grads
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

function train(experiment, params)
    local iters = params.iters
    local useCuda = params.useCuda or false
    local numGPUs = params.numGPUs or 1

    local model = experiment.model
    local criterion = experiment.criterion
    local dataset = experiment.dataset
    local group = experiment.group
    local optimizer = experiment.optimizer
    local batchSize = experiment.batchSize

    if useCuda then 
        require 'cunn'
        require 'cutorch'
        cudaInput = torch.CudaTensor()
        cudaOutput = torch.CudaTensor()

        cudaInputValidation = torch.CudaTensor()
        cudaOutputValidation = torch.CudaTensor()

        model = model:cuda()
        criterion = criterion:cuda()
        model = makeDataParallel(model, numGPUs)
    end
  
    parameters, grads = model:getParameters()

    train_costs = {}
    validation_costs = {}
    cost_average = nil

    validation = dataset:minibatch("validate", experiment.validationSize)
    validation_cost = -1

    for i = 1, iters do
        batch = dataset:minibatch(experiment.group, experiment.batchSize)

        local startTime = sys.clock()

        if useCuda then
            cudaInput:resize(batch.input:size()):copy(batch.input:float())
            cudaOutput:resize(batch.output:size()):copy(batch.output:float())
            cost, grad = eval(model, cudaInput, cudaOutput, criterion)
        else
            cost, grad = eval(model, batch.input, batch.output, criterion)
        end

        local iterTime = sys.clock() - startTime

        if cost_average == nil then cost_average = cost end
        cost_average = .95*cost_average + .05*cost

        if i % 10 == 0 then
            if i % 2000 == 0 then 
                validation_cost, _ = eval(model, validation.input, 
                    validation.output, criterion)
                print("training", cost_average, "validation", 
                    validation_cost)
                table.insert(validation_costs, validation_cost)
            else
                print("training", cost_average, "(samples per second "..batchSize/iterTime ..")")
            end
            table.insert(train_costs, cost)
        end

        optimizer:step(parameters, grad)
    end

    return train_costs, validation_costs
end
