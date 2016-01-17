require 'paths'
require 'torch'
require 'dataloader'
Threads = require 'threads'
Threads.serialization("threads.sharedserialize")

Dataset = {}
Dataset.root = "/home/ubuntu/ebs_disk"
Dataset.directory = "train"

function prepare_data_loaders(number_loaders)
    number_loaders = number_loaders or 32
    if number_loaders > 1 then
        data_loaders = Threads(
            number_loaders,
            function()
                paths.dofile('dataloader.lua')
            end
        )
    else
        data_loaders = {}
        function data_loaders:addjob(f1, f2) f2(f1()) end
        function data_loaders:synchronize() end
    end
end

prepare_data_loaders()

function Dataset:generate_random_filename()
    if not self.initialized then self:init() end

    local games = self.games
    local game = games[math.random(1, #games)]
    local random_move = math.random(1, game.size)

    return game.name .. "/" .. random_move
end

function Dataset:new(dict)
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

function Dataset:init()
    --read index file with paths to games and move counts
    self.games = {}
    local path = self.root .. "/" .. self.directory
    print("initializing dataset", path)
    local file = path .. "_game_counts.txt"
    local counts = io.open(file)

    if counts == nil then 
        local command = "sh count_game_moves.sh " .. path .. " > " .. file
        print (command)
        io.popen(command):close()
        counts = io.open(file)
        assert(counts ~= nil, "unable to find counts (probably bad dataset path)")
    end 

    for line in counts:lines() do
        local r = line:split("\t")
        local game = { name = r[1], size = tonumber(r[2]) } 

        --some games have no moves and we want to remove those
        if game.size > 0 then 
            table.insert(self.games, game) 
        end
    end 
    counts:close()
    self.initialized = true
end

function make_minibatch(dataset, n)
    local result
    queue_on_minibatch(function(x) result = x end, dataset, n)
    do_queued_tasks()
    return result
end

function queue_on_minibatch(f, dataset, n)
    if dataset.initialized ~= true then dataset:init() end
    local filenames = {}
    for i = 1, n do
        table.insert(filenames, dataset:generate_random_filename())
    end
    data_loaders:addjob(
        function() return load_minibatch(filenames) end,
        f
    )
end

function do_queued_tasks()
    data_loaders:synchronize()
end
