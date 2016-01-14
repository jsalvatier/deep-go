require 'data'
require 'torch'

GoDataset = Dataset:new()
GoDataset.output_dimensions = {}

SIZE = 19

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

INPUT = {}
INPUT.stone = 1
INPUT.liberties = INPUT.stone+1
INPUT.liberties_after = INPUT.liberties+1
INPUT.kills = INPUT.liberties_after+2
INPUT.age = INPUT.kills+2
INPUT.ladder = INPUT.age+1
INPUT.total = INPUT.ladder+1

-- this is used by makedata.lua when writing data
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


GoDataset.input_dimensions = {TOTAL, SIZE, SIZE}

function transform(grid)
    -- eventually this should do random rotation and reflection, but for now lets just skip it
    return grid
end

function expand_to(n, x)
    return x:repeatTensor(n, 1, 1)
end

function GoDataset:preprocess_fast(data)
    local input = torch.DoubleTensor(torch.LongStorage(GoDataset.input_dimensions)):zero()
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

function GoDataset:preprocess(data)
    ---takes a torch object of data and turns it into the actual dataset we're going to train on
    
    --if the data is flat, use the fast preprocessor:
    if data.flat then return GoDataset.preprocess_fast(self, data) end

    --otherwise use the normal preprocessor
    local input = torch.DoubleTensor(torch.LongStorage(self.input_dimensions)):zero()
    local player = data.move.player
    local stones = data.stones
    local ages = data.age
    local liberties = data.liberties
    local liberties_after = (data.liberties_after or data.liberties_after_move)[player]
    local kills = data.kills[player]
    local ladder
    if data.ladders ~= nil then ladder = data.ladders[player] end
    local rank = data.ranks[player] or 1

    -- if white plays next, we will reverse white and black
    function swap_if_white(p)
        if player == 2 and p >= 1 then
            return 3-p
        else
            return p
        end
    end

    --we randomly rotate and reflect the board
    reflect_x = math.random() > 0.5
    reflect_y = math.random() > 0.5
    reflect_diagonal = math.random() > 0.5
    local function transform(t)
        if reflect_x then t[1] = 20 - t[1] end
        if reflect_y then t[2] = 20 - t[2] end
        if reflect_diaognal then t[1], t[2] = t[2], t[1] end
        return t
    end

    -- for now suppress the randm reflection...
    local function transform(t)
        return t
    end



    --mark stones by 1 on different layers of the input array
    for i = 1, SIZE do
        for j = 1, SIZE do
            local stone = stones[transform{i, j}]
            local liberty = liberties[transform{i, j}]
            local age = ages[transform{i,j}]
            local liberty_after = liberties_after[transform{i, j}]
            local kill = kills[transform{i, j}]
            local ladder = 0
            if ladders ~= nil then ladder = ladders[transform{i, j}] end

            if stone > 0 then
                input[{STONE+swap_if_white(stone), i, j}] = 1
            else
                input[{STONE, i, j}] = 1
            end

            if liberty >= 1 then
                input[{LIBERTIES+math.min(liberty, 4)-1,i,j}] = 1
            end

            if stone == 0 then
                input[{LIBERTIES_AFTER+math.min(liberty_after, 6),i,j}] = 1
            end

            if kill >= 1 then
                input[{KILL+math.min(kill, 7)-1, i, j}] = 1
            end

            if age >= 1 and age <= 5 then
                input[{AGE+age-1,i,j}] = 1
            end

            if ladder > 0 then
                input[{LADDER,i,j}] = 1
            end
        end
    end

    input[RANK+rank] = torch.ones(SIZE, SIZE)

    local transformed_move = transform{data.move.x, data.move.y}

    --mark move actually made
    local output = SIZE * (transformed_move[1] - 1) + transformed_move[2]
    return {input=input, output=output}
end

ToyDataset = GoDataset:new()

ToyDataset.directories = {toy='toy', validate='toy'}
ToyDataset.root = "data"
