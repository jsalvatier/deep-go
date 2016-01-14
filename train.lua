require 'godata'

function eval(model, inputs, targets, criterion)
    grads:zero()
    preds = model:forward(inputs)

    cost = criterion:forward(preds, targets)
    df_do = criterion:backward(preds, targets)
    model:backward(inputs, df_do)
    return cost, grads
end

function train(model, criterion, batchSize, iters, optimizer, useCuda, dataset, group, validationSize)

    if useCuda then 
        require 'cunn'
        require 'cutorch'
        cudaInput = torch.CudaTensor()
        cudaOutput = torch.CudaTensor()

        cudaInputValidation = torch.CudaTensor()
        cudaOutputValidation = torch.CudaTensor()

        model = model:cuda()
        criterion = criterion:cuda()
    end
  
    parameters, grads = model:getParameters()

    train_costs = {}
    validation_costs = {}
    cost_average = 5

    validation = dataset:minibatch("validate", validationSize)
    validation_cost = -1

    for i = 1, iters do
        batch = dataset:minibatch(group, batchSize)

        if useCuda then
            cudaInput:resize(batch.input:size()):copy(batch.input:float())
            cudaOutput:resize(batch.output:size()):copy(batch.output:float())
            cost, grad = eval(model, cudaInput, cudaOutput, criterion)
        else
            cost, grad = eval(model, batch.input, batch.output, criterion)
        end

        cost_average = .95*cost_average + .05*cost

        if i % 10 == 0 then
            if i % 2000 == 0 then 
                validation_cost, _ = eval(model, validation.input, 
                    validation.output, criterion)
                print("training", cost_average, "validation", 
                    validation_cost)
                table.insert(validation_costs, validation_cost)
            else
                print("training", cost_average)
            end
            table.insert(train_costs, cost)
        end

        optimizer:step(parameters, grad)
    end

    return train_costs, validation_costs
end
