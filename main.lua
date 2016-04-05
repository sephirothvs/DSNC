require 'xlua'
require 'optim'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'

require 'DeepSNC.lua'
require 'dknn.lua'
dofile 'DataLoader.lua'
dofile 'util.lua'
-- ==============
-- commandline
-- ==============
local function commandLine()
opt = lapp [[
	-s, --save              (default "logs")    subdirectory for logs
	-b, --batchSize         (default 100)       batch size
	-r, --learningRate      (default 5e-4)       learning rate
	-d, --dataSet           (default mnist)     name of dataSet
	-i, --device			(default 1)	    	gpu id
	-n, --network			(default lenet5)    model name
	--max_epoch             (default 100)       maximum number of epoches
	--validate              (default 0.2)       validation ratio
	--dnum					(default 100)		data compression size
	--gamma 				(default 1.)  		gamma value in the loss function
	--dim					(default 50)		feature dimentions
	--lambda				(default 1e-5) 		penalty term
    --lambda2               (default 1e-5)      penalty term 2
	--dropout               (default 0)			dropout rate
	--dSNC					(default 1)			1: DSNC  ; 0: softmax
]]
return opt
end

-- ==================
-- main function
-- ==================
local function main()
	torch.manualSeed(345)
	cutorch.manualSeed(345)
	local opt  = commandLine()
	cutorch.setDevice(opt.device)
	data = load_data(opt)
	print(opt)
	-- adjust opt
	local nTrain = data.xr:size(1)
	opt.nBatches = math.ceil(nTrain/opt.batchSize)
	opt.optim_config = {
		learningRate = opt.learningRate,
		alpha = 0.99,
		epsilon = 1e-8
	}
	print(opt.optim_config)
	opt.optimizer = optim.rmsprop
	-- load model
	local model = dofile(opt.network..'.lua')
    model:cuda()
	local confusion = optim.ConfusionMatrix(opt.nClass)
	
	if opt.dSNC == 1 then
		local indx  = initialize(data.yr, opt.nClass)
    	local sets  = data.xr:index(1, indx):sub(1, opt.dnum):cuda()
    	data.xz     = torch.CudaTensor():resize(opt.dnum, opt.dim):copy(model:forward(sets))
    	data.yz     = torch.CudaTensor():resize(opt.dnum):copy(data.yr:index(1, indx):sub(1, opt.dnum))
    	sets        = nil
		model:add(nn.DeepSNC(data.xz, data.yz, opt):cuda())
	else
		criterion = nn.CrossEntropyCriterion():cuda()
	end
	print(data)
	local W, grad   = model:getParameters()
	print(model)
	
	paths.mkdir(opt.save)
	testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
	testLogger:setNames{'% 1nn_err (train set)', '% 1nn_err (valid set)', '% 1nn_err (test set)'}
	testLoggere5showPlot = false	

	local report    = reportErr(data, model, opt, confusion)
	for t = 1, opt.max_epoch do
		train(model, criterion, W, grad, data, opt)
		local flag = report(t)
		collectgarbage()
	end
end

main()
