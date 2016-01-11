Dataset = {}

Dataset.default_group = 'train'
Dataset.default_minibatch = 32
Dataset.root = "data/"
Dataset.directories = {train='train', test='test', validate='validate'}
Dataset.input_dimensions = {2, 19, 19}
Dataset.output_dimensions = {10}

function Dataset:init()
    self.files = {}
    self._initialized = true
    for group, directory in pairs(self.directories) do
        self.files[group] = {}

        local filelist = io.popen("find "..self.root..directory.." -type f"):lines()
        for file in filelist do
            table.insert(self.files[group], file)
        end
    end
end

function Dataset:load_random_datum(group)
    if self._initialized ~= true then self:init() end
    n = #self.files[group]
    index = math.random(1, n)
    return self:load_and_preprocess(self.files[index])
end

function Dataset:load_and_preprocess(filename)
    if self._initialized ~= true then self:init() end
    return self:preprocess(self:load_file(filename))
end

function Dataset:load_file(filename)
    if self._initialized ~= true then self:init() end
    return torch.load(filename)
end

function Dataset:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    return o
end

function Dataset:preprocess(data)
    return data
end

function Dataset:new_inputs_outputs(size)
    local input_dimensions = {}
    table.insert(input_dimensions, size)
    for _, dimension in pairs(self.input_dimensions) do
        table.insert(input_dimensions, dimension)
    end
    local output_dimensions = {}
    table.insert(output_dimensions, size)
    for _, dimension in pairs(self.output_dimensions) do
        table.insert(output_dimensions, dimension)
    end

    local inputs = torch.DoubleTensor(torch.LongStorage(input_dimensions)):zero()
    local outputs = torch.DoubleTensor(torch.LongStorage(output_dimensions)):zero()

    return inputs, outputs
end

function Dataset:minibatch(group, size)
    if self._initialized ~= true then self:init() end
    size = size or self.default_minibatch
    local inputs, outputs = self:new_inputs_outputs(size)
    for i = 1, size do
        local data = self:load_random_datum(group)
        inputs[i] = data.input
        outputs[i] = data.output
    end
    return {input=inputs, output=outputs}
end

function Dataset:load_all(group)
    if self._initialized ~= true then self:init() end
    local files = self.files[group]
    local size = #files
    local inputs, outputs = self:new_inputs_outputs(size)
    
    for i = 1, size do
        local data = self:load_and_preprocess(files[i])
        inputs[i] = data.input
        outputs[i] = data.output
    end
    return {input=inputs, output=outputs}
end

GoDataset = Dataset:new()
GoDataset.root = "data/"

function GoDataset:preprocess(data)
    ---takes a torch object of data and turns it into the actual dataset we're going to train on
    if self._initialized ~= true then self:init() end
    input = torch.DoubleTensor(data.board:size()):zero()
    local board = data.board[1]

    --mark stones by 1 on different layers of the input array
    for i = 1, 19 do
        for j = 1, 19 do
            if board[i][j] > 0 then input[{board[i][j], i, j}] = 1 end
        end
    end

    --mark move actually made
    output = 19 * (data.move.x - 1) + (data.move.y - 1)
    return {input=input, output=output}
end
