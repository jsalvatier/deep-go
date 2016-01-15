require 'data'
require 'dataloader'

function eval(model, grads, inputs, targets, criterion)
    grads:zero()
    local preds = model:forward(inputs)

    local cost = criterion:forward(preds, targets)
    local df_do = criterion:backward(preds, targets)
    model:backward(inputs, df_do)
    return cost, grads
end

function eval_validation(experiment, input, targets)
    local model = experiment.model
    local criterion = experiment.criterion
    local batchSize = experiment.batchSize
    local validation_size = (#input)[1]

    local sum_costs = 0

    for i = 1,validation_size/batchSize do
        range = {(i-1)*batchSize + 1, math.min(i*batchSize, validation_size)}
        batch = input[{range}]
        batch_targets = targets[{range}]

        local preds = model:forward(batch)
        local cost = criterion:forward(preds, batch_targets)

        local size = range[2] - range[1]
        sum_costs = sum_costs + cost*size
    end

    return sum_costs/validation_size
end

function train(experiment, params)
    local iters = params.iters

    local useCuda = experiment.useCuda
    local model = experiment.model
    local criterion = experiment.criterion
    local datasets = experiment.datasets
    local group = experiment.group
    local optimizer = experiment.optimizer
    local batchSize = experiment.batchSize
    local validationSize = experiment.validationSize
    local parameters = experiment.modelParameters
    local grads = experiment.grads


    local validationInput, validationOutput
    local function set_validation_data(minibatch)
        validationInput = minibatch.input
        validationOutput = minibatch.output
    end
    queue_on_minibatch(set_validation_data, datasets.validation, validationSize)
    do_queued_tasks()
    local validation_cost = -1

    local cudaInput, cudaOutput

    if useCuda then 
        require 'cunn'
        require 'cutorch'
        cudaInput = torch.CudaTensor()
        cudaOutput = torch.CudaTensor()

        local cudaInputValidation = torch.CudaTensor()
        local cudaOutputValidation = torch.CudaTensor()

        cudaInputValidation:resize(validationInput:size()):copy(validationInput:float())
        cudaOutputValidation:resize(validationOutput:size()):copy(validationOutput:float()) 

        validationInput = cudaInputValidation
        validationOutput = cudaOutputValidation
    end
  
    local train_costs = {}
    local cost_average = nil

    local function train_iter(train_set)
        local startTime = sys.clock()
        local input = train_set.input
        local output = train_set.output

        if useCuda then
            cudaInput:resize(input:size()):copy(input:float())
            cudaOutput:resize(output:size()):copy(output:float())

            input = cudaInput
            output = cudaOutput
        end

        if pcall(function() eval(model, grads, input, output, criterion) end) ~= true then
            global_input = input
            global_output = output
        end

        local train_cost, _ = eval(model, grads, input, output, criterion)
        
        local iterTime = sys.clock() - startTime
        if cost_average == nil then cost_average = train_cost end
        cost_average = .95*cost_average + .05*train_cost
        
        experiment.iterations = experiment.iterations + 1

        if experiment.iterations % 10 == 0 then
            if experiment.iterations % experiment.validation_interval == 0 then 
                validation_cost, _ = eval_validation(experiment, validationInput, validationOutput)
                print("training", cost_average, "validation", 
                    validation_cost)
                table.insert(experiment.validation_costs, validation_cost)
     
                experiment:save()
            else
                print("training", cost_average, "(samples per second "..batchSize/iterTime ..")")
            end
            table.insert(train_costs, cost_average)
        end

        optimizer:step(parameters, grads)
    end

    for i = 1, iters do queue_on_minibatch(train_iter, datasets.train, batchSize) end
    do_queued_tasks()

    return train_costs, validation_costs
end
