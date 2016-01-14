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

GoDataset.input_dimensions = {TOTAL, SIZE, SIZE}

function GoDataset:preprocess(data)
    ---takes a torch object of data and turns it into the actual dataset we're going to train on
    if self._initialized ~= true then self:init() end
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
    function transform(t)
        if reflect_x then t[1] = 20 - t[1] end
        if reflect_y then t[2] = 20 - t[2] end
        if reflect_diaognal then t[1], t[2] = t[2], t[1] end
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
ToyDataset.root = "data/"
