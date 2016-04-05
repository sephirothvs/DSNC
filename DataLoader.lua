dofile 'provider.lua'
function load_data(opt)
    local mnist     = require 'mnist'
    local trainData = mnist.traindataset()
    local testData  = mnist.testdataset()
    local data      = {}
    data['xr']      = trainData.data:float()
    data['xe']      = testData.data:float()
    data['yr']      = trainData.label + 1 
    data['ye']      = testData.label + 1 
	-- reshape data n * d * w*h
	data.xr     = data.xr:reshape(60000, 1, 28, 28) 
    data.xe     = data.xe:reshape(10000, 1, 28, 28) 
    -- shuffle data
    local shuffle_idx = torch.randperm(data.xr:size(1),'torch.LongTensor')
    data.xr           = data.xr:index(1,shuffle_idx)
    data.yr           = data.yr:index(1,shuffle_idx)
    -- normalization
    local x_max = data.xr:max()
    data.xr:div(x_max)
    data.xe:div(x_max)
	-- validationset
	local nValid = math.floor(data.xr:size(1) * opt.validate)
    local nTrain = data.xr:size(1) - nValid
    data['xv']   = data.xr:sub(nTrain+1,data.xr:size(1))
    data['yv']   = data.yr:sub(nTrain+1,data.xr:size(1))
    data['xr']   = data.xr:sub(1,nTrain)
    data['yr']   = data.yr:sub(1,nTrain)
    opt.nClass   = 10
    return data
end