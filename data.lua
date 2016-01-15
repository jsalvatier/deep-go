require 'paths'
require 'torch'
Threads = require 'threads'
Threads.serialization("threads.sharedserialize")

Dataset = {}
Dataset.root = "/home/ubuntu/ebs_disk"
Dataset.directory = "train"

function prepare_data_loaders(number_loaders)
    number_loaders = number_loaders or 1
    if number_loaders > 1 then
        data_loaders = Threads(
            number_loaders,
            function()
                paths.dofile('data.lua')
                paths.dofile('godata.lua')
            end
        )
    else
        data_loaders = {}
        function data_loaders:addjob(f1, f2) f2(f1()) end
        function data_loaders:synchronize() end
    end
end

prepare_data_loaders()

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

function Dataset:generate_random_filename()
    if not self.initialized then self:init() end

    local games = self.games
    local game = games[math.random(1, #games)]
    local random_move = math.random(1, game.size)

    return game.name .. "/" .. random_move
end

function load_and_preprocess(filename)
    return preprocess(torch.load(filename))
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

function minibatch(dataset, size)
    if dataset.initialized ~= true then dataset:init() end
    local inputs, outputs = empty_minibatch_io(size)
    for i = 1, size do
        local filename = dataset:generate_random_filename()
        local data = load_and_preprocess(filename)
        inputs[i] = data.input
        outputs[i] = data.output
    end
    return {input=inputs, output=outputs}
end

--definitions of channels
--all features are from the perspective of black
STONE=1 --1=black, 2=white, 3=empty
LIBERTIES=STONE+3 --1 liberty, 2 liberties, ..., 4 or more liberties
LIBERTIES_AFTER=LIBERTIES+4 -- 0, 1, 2, ..., 6 or more
KILL=LIBERTIES_AFTER+7 --1, 2, ..., 7 or more
AGE=KILL+7 --1, 2, ..., 5
LADDER=AGE+5
RANK=LADDER+1 --1d, 2d, ..., 9d

TOTAL = RANK+9
SIZE = 19

input_dimensions = {TOTAL, SIZE, SIZE}
output_dimensions = {}

INPUT = {}
INPUT.stone = 1
INPUT.liberties = INPUT.stone+1
INPUT.liberties_after = INPUT.liberties+1
INPUT.kills = INPUT.liberties_after+2
INPUT.age = INPUT.kills+2
INPUT.ladder = INPUT.age+1
INPUT.total = INPUT.ladder+1

-- this is used by makedata.lua when preparing the data (that will be read here)
function flatten_data(data)
    local result = torch.ByteTensor(INPUT.total, SIZE, SIZE)
    result[INPUT.stone] = data.stones
    result[INPUT.liberties] = data.liberties
    result[{{INPUT.liberties_after, INPUT.liberties_after+1}}] = data.liberties_after or data.liberties_after_move
    result[{{INPUT.kills, INPUT.kills+1}}] = data.kills
    result[INPUT.age] = data.age or data.ages
    result[{{INPUT.ladder, INPUT.ladder+1}}] = data.ladders or torch.zeros(2, SIZE, SIZE)
    return {input=result, move=data.move, ranks=data.ranks, flat=true}
end

function transform(grid)
    -- eventually this should do random rotation and reflection, but for now lets just skip it
    return grid
end

function expand_to(n, x)
    return x:repeatTensor(n, 1, 1)
end

function preprocess(data)
    local input = torch.DoubleTensor(torch.LongStorage(input_dimensions)):zero()
    local raw = data.input
    raw = transform(raw)
    local player = data.move.player


    input[STONE] = torch.eq(raw[INPUT.stone], 0)
    input[STONE+1] = torch.eq(raw[INPUT.stone], player)
    input[STONE+2] = torch.eq(raw[INPUT.stone], 3-player)


    local numbers = torch.ByteTensor(7, SIZE, SIZE)
    for i = 1, 7 do
        numbers[i] = torch.ones(SIZE, SIZE) * i
    end

    local raw_liberties = raw[INPUT.liberties]
    input[{{LIBERTIES, LIBERTIES+2}}][{{1, 3}}] = torch.eq(expand_to(3,raw_liberties), numbers[{{1, 3}}])
    input[LIBERTIES+3] = torch.ge(raw_liberties, 4)


    local raw_liberties_after = raw[INPUT.liberties_after + player - 1]
    input[LIBERTIES_AFTER]:addcmul(input[STONE], torch.eq(raw_liberties_after, 0):double())
    input[{{LIBERTIES_AFTER+1, LIBERTIES_AFTER+5}}] = torch.eq(expand_to(5,raw_liberties_after), numbers[{{1, 5}}])
    input[LIBERTIES_AFTER+6] = torch.ge(raw_liberties_after, 6)

    local raw_kills = raw[INPUT.kills + player - 1]
    input[{{KILL, KILL+5}}] = torch.eq(expand_to(6,raw_kills), numbers[{{1,6}}])
    input[KILL+6] = torch.ge(raw_kills, 7)

    local raw_age = raw[INPUT.age]
    input[{{AGE, AGE+4}}] = torch.eq(expand_to(5,raw_age), numbers[{{1, 5}}])

    local raw_ladder = raw[INPUT.ladder + player - 1]
    input[LADDER] = torch.ge(raw_ladder, 1)

    input[RANK+data.ranks[player]] = numbers[1]

    local output = SIZE * (data.move.x - 1) + data.move.y
    return {input=input, output=output}

end

function empty_minibatch_io(size)
    local minibatch_input_dimensions = {}
    table.insert(minibatch_input_dimensions, size)
    for _, dimension in pairs(input_dimensions) do
        table.insert(minibatch_input_dimensions, dimension)
    end

    local minibatch_output_dimensions = {}
    table.insert(minibatch_output_dimensions, size)
    for _, dimension in pairs(output_dimensions) do
        table.insert(minibatch_output_dimensions, dimension)
    end

    local inputs = torch.DoubleTensor(torch.LongStorage(minibatch_input_dimensions)):zero()
    local outputs = torch.DoubleTensor(torch.LongStorage(minibatch_output_dimensions)):zero()

    return inputs, outputs
end
