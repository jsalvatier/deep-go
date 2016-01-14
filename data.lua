Dataset = {}

Dataset.default_group = 'train'
Dataset.default_minibatch = 32
Dataset.root = "~/ebs_disk"
Dataset.directories = {train='train', test='test', validate='validate'}

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
    if self._initialized ~= true then
        print("initializing data set")
        self:init()
    end
    n = #self.files[group]
    index = math.random(1, n)
    return self:load_and_preprocess(self.files[group][index])
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
