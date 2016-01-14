require 'godata'

function eval(model, grads, inputs, targets, criterion)
    grads:zero()
    local preds = model:forward(inputs)

    local cost = criterion:forward(preds, targets)
    local df_do = criterion:backward(preds, targets)
    model:backward(inputs, df_do)
    return cost, grads
end

function train(experiment, params)
    local iters = params.iters

    local useCuda = experiment.useCuda
    local model = experiment.model
    local criterion = experiment.criterion
    local dataset = experiment.dataset
    local group = experiment.group
    local optimizer = experiment.optimizer
    local batchSize = experiment.batchSize
    local validationSize = experiment.validationSize
    local parameters = experiment.modelParameters
    local grads = experiment.grads

    local validation_set = dataset:minibatch("validate", validationSize)
    local validation_cost = -1
    local validationInput = validation_set.input
    local validationOutput = validation_set.output 

    if useCuda then 
        require 'cunn'
        require 'cutorch'
        local cudaInput = torch.CudaTensor()
        local cudaOutput = torch.CudaTensor()

        local cudaInputValidation = torch.CudaTensor()
        local cudaOutputValidation = torch.CudaTensor()

        cudaInputValidation:resize(validationInput:size()):copy(validationInput:float())
        cudaOutputValidation:resize(validationOutput:size()):copy(validationOutput:float()) 

        validationInput = cudaInputValidation
        validationOutput = cudaOutputValidation
    end
  
    local train_costs = {}
    local validation_costs = {}
    local cost_average = nil

    for i = 1, iters do
        local startTime = sys.clock()
        local train_set = dataset:minibatch(group, batchSize)
        local input = train_set.input
        local output = train_set.output

        if useCuda then
            cudaInput:resize(input:size()):copy(input:float())
            cudaOutput:resize(output:size()):copy(output:float())

            input = cudaInput
            output = cudaOutput
        end

        local train_cost, _ = eval(model, grads, input, output, criterion)
        
        local iterTime = sys.clock() - startTime
        if cost_average == nil then cost_average = train_cost end
        cost_average = .95*cost_average + .05*train_cost

        if i % 10 == 0 then
            if i % 2000 == 0 then 
                validation_cost, _ = eval(model, validationInput, 
                    validationOutput, criterion)
                print("training", cost_average, "validation", 
                    validation_cost)
                table.insert(validation_costs, validation_cost)
     
                experiment:save()
            else
                print("training", cost_average, "(samples per second "..batchSize/iterTime ..")")
            end
            table.insert(train_costs, cost_average)
        end

        optimizer:step(parameters, grads)
    end

    return train_costs, validation_costs
end
