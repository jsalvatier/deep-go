Threads = require 'threads'
Threads.serialization("threads.sharedserialize")

Dataset = {}

Dataset.default_group = 'train'
Dataset.default_minibatch = 32
Dataset.root = "/home/ubuntu/ebs_disk"
Dataset.directories = {train='train', test='test', validate='validation'}
Dataset.num_threads = 32

function Dataset:init()
    self.game_names = {}
    self.game_sizes = {}
    self._initialized = true
    -- this is an extremely ugly hack
    -- the problem is that metatables don't get sent to threads...
    -- previously preprocess might live on the metatable
    self.preprocess = self.preprocess
    do
        if self.num_threads > 1 then
            local calling_dataset = self
            self.thread_pool = Threads(
                self.num_threads,
                function() 
                    dataset = calling_dataset  
                    -- jesus, this is ugly... (I started writing better code, but it doesn't work with Threads...)
                    dofile 'godata.lua'
                end
            )
        else
            self.thread_pool = {}
            local dataset = self
            function self.thread_pool:addjob(f1, f2) f2(f1()) end
            function self.thread_pool:synchronize() end
        end
    end



    self.games = {}
    for group, directory in pairs(self.directories) do
        local counts = io.open(self.root .. "/" .. directory .. "_game_counts.txt")

        for line in counts:lines() do
            local r = line:split("\t")
            print (r)
            game = { name = r[1], size = tonumber(r[2]) } 

            if game.size > 0 then 
                table.insert(self.games, game) 
            end

        end 
        counts:close()

        
    end
end

function Dataset:generate_random_filename(group)
    local games = self.games

    local game = games[math.random(1, #games)]
    local random_move = math.random(1, game.size)

    return game.name .. "/" .. random_move
end

function Dataset:load_random_datum(group)
    if self._initialized ~= true then
        print("initializing data set")
        self:init()
    end
    return self:load_and_preprocess(self:generate_random_filename(group))
end

function Dataset:load_and_preprocess(filename)
    return self:preprocess(self:load_file(filename))
end

function Dataset:load_file(filename)
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
    if self.num_threads == 1 then
        dataset = self
    end
    for i = 1, size do
        local file = self:generate_random_filename(group)
        self.thread_pool:addjob(function()
            local data = torch.load(file)
            return dataset:preprocess(data)
        end,
        function(data)
            inputs[i] = data.input
            outputs[i] = data.output
        end
        )
    end
    self.thread_pool:synchronize()
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
