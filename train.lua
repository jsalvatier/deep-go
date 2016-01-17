require 'data'
require 'dataloader'

function eval(experiment, inputs, targets)
    local criterion = experiment.criterion
    local model = experiment.model
    local grads = experiment.grads
    inputs = experiment:prepare_data(inputs)
    targets = experiment:prepare_data(targets)
    if grads == nil then print(experiment) end
    grads:zero()

    local preds = model:forward(inputs)
    local cost = criterion:forward(preds, targets)
    local output_grads = criterion:backward(preds, targets)
    model:backward(inputs, output_grads)

    return cost, grads
end

function cost_and_accuracy(experiment, inputs, targets)
    local model = experiment.model
    local criterion = experiment.criterion
    local batchSize = experiment.batchSize
    local validation_size = (#inputs)[1]
    inputs = experiment:prepare_data(inputs)
    targets = experiment:prepare_data(targets)

    local sum_costs = 0
    local sum_errors = 0

    for i = 1,validation_size/batchSize do
        range = {(i-1)*batchSize + 1, math.min(i*batchSize, validation_size)}
        batch = inputs[{range}]
        batch_targets = targets[{range}]

        local preds = model:forward(batch)
        local max, top_preds = preds:max(2)

        local long_batch_targets = batch_targets
        if not experiment.useCuda then 
            long_batch_targets = batch_targets:long()
        end 

        local errors = torch.sum(top_preds:ne(long_batch_targets))
        local cost = criterion:forward(preds, batch_targets)

        local size = range[2] - range[1] + 1
        sum_costs = sum_costs + cost*size
        sum_errors = sum_errors + errors
    end

    return sum_costs/validation_size, (1-sum_errors/validation_size)
end

function validate(experiment, dataset, size)
    dataset = dataset or experiment.datasets
    size = size or experiment.validationSize
    local minibatch = make_minibatch(dataset.validation, size)
    return cost_and_accuracy(experiment, minibatch.input, minibatch.output)
end


function train(experiment, params)
    local iters = params.iters

    local datasets = experiment.datasets
    local optimizer = experiment.optimizer
    local batchSize = experiment.batchSize
    local validationSize = experiment.validationSize
    local parameters = experiment.modelParameters
    local grads = experiment.grads

    local validation_minibatch = make_minibatch(datasets.validation, validationSize)

    local train_costs = {}
    local cost_average = nil

    local total_start_time = sys.clock()

    local function train_iter(train_set)
        local iter_start_time = sys.clock()
        local input = train_set.input
        local output = train_set.output

        local train_cost, _ = eval(experiment, input, output)
        
        local iter_time = sys.clock() - iter_start_time
        if cost_average == nil then cost_average = train_cost end
        cost_average = .95*cost_average + .05*train_cost
        
        experiment.iterations = experiment.iterations + 1

        if experiment.iterations % 10 == 0 then
            print("training cost at iteration "..experiment.iterations..": "..cost_average, " (samples per second "..batchSize/iter_time..")")
            table.insert(train_costs, cost_average)
        end
        if experiment.iterations % experiment.validation_interval == 0 then 
            local validation_input, validation_output = validation_minibatch.input, validation_minibatch.output
            local validation_cost, validation_accuracy = cost_and_accuracy(experiment, validation_input, validation_output)
            print("validation at iteration "..experiment.iterations..": cost="..validation_cost..", accuracy="..validation_accuracy)
            table.insert(experiment.validation_costs, validation_cost)
            experiment:save()
        end

        optimizer:step(parameters, grads)
    end

    for i = 1, iters do queue_on_minibatch(train_iter, datasets.train, batchSize) end
    do_queued_tasks()

    local total_time = sys.clock() - total_start_time

    print("total samples per second", batchSize * iters / total_time)

    return train_costs, validation_costs
end
