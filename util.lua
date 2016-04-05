function copy2cpu(tdata)
    local tmp = torch.Tensor()
    tmp:resize(tdata:size()):copy(tdata)
    return tmp
end

function initialize(y,class)
    local save = torch.Tensor(class)
    for i =1,class do
        local _, ind = torch.eq(y, i):sort()
        save[i] = ind[-1]
    end 
    local perm = torch.rand(y:size(1))
    for i = 1, class do
        perm[save[i]] = -100
    end 
    _, res = torch.sort(perm)
    return res 
end

function feedforward(model, data, opt)
    local res = torch.Tensor()
    res:resize(data:size(1), opt.dim):fill(0)
    model:evaluate()
    local N = data:size(1)
    local dataGPU = torch.CudaTensor()
    for k = 1, N, opt.batchSize do
        local idx       = math.min(k+opt.batchSize-1, N)
        local inputs    = data:sub(k, idx)
        dataGPU:resize(inputs:size()):copy(inputs)
        local outputs   = lastOutput(model, dataGPU)
		local tmp = copy2cpu(outputs)
        setidx = torch.range(k, idx):type('torch.LongTensor')
        res:indexCopy(1, setidx, tmp)
    end
    return res
end

function lastOutput(model, data)
	model:evaluate()
    local m = model:size()
    local input = data
    local output = input
    for i = 1, m-1 do
        output = model:get(i):forward(input)
        input = output
    end
    return output
end

function FinalOutput(model, data, label)
    local m = model:size()
    local input = data
    local output = input
    for i = 1, m-1 do
        output = model:get(i):forward(input)
        input = output
    end
    output = model:get(m):forward(input, label)
    return output
end


function train(model, criterion, W, grad, data, opt)
    model:training() 
    local inputs_gpu  = torch.CudaTensor()
    local targets_gpu = torch.CudaTensor()
    local nTrain      = data.xr:size(1)
    local shuffle_idx = torch.randperm(nTrain, 'torch.LongTensor')
    data.xr           = data.xr:index(1, shuffle_idx)
    data.yr           = data.yr:index(1, shuffle_idx)
     for t = 1, nTrain, opt.batchSize do
        local idx       = math.min(t+opt.batchSize-1, nTrain)
        local inputs    = data.xr:sub(t, idx)
        local targets   = data.yr:sub(t, idx)
        inputs_gpu:resize(inputs:size()):copy(inputs)
        targets_gpu:resize(targets:size()):copy(targets)
        function feval(x)
            assert(x==W)
            grad:zero()
			if opt.dSNC == 1 then
				local f = FinalOutput(model, inputs_gpu, targets_gpu)
				model:backward(inputs_gpu, targest_gpu)
				f = f/opt.batchSize
			elseif opt.dSNC ==0 then
				local outputs	= model:forward(inputs_gpu)
				local f 		= criterion:forward(outputs, targets_gpu)
				local df_dw 	= criterion:backward(outputs, targets_gpu)
				model:backward(inputs_gpu, df_dw)
				f = f/opt.batchSize
			else
				print('**** error for opt.dSNC (0 or 1) ****')
			end
            return f, grad
        end 
        opt.optimizer(feval,  W, opt.optim_config)
    end 
end


function evaluation(suffix, data, model, batchSize, confusion)
    if suffix ~= 'r' and suffix ~= 'e' and suffix ~= 'v' then
        error('Unrecognized dataset specified')
    end
    model:evaluate()
    local N     = data['x' .. suffix]:size(1)
    local err   = 0
    local inputs_gpu = torch.CudaTensor()
    local targets_gpu = torch.CudaTensor()
    for k = 1, N, batchSize do
        local idx         = math.min(k+batchSize-1,N)
        local inputs      = data['x' .. suffix]:sub(k,idx)
        local targets     = data['y' .. suffix]:sub(k,idx)
        inputs_gpu:resize(inputs:size()):copy(inputs)
        targets_gpu:resize(targets:size()):copy(targets)
		if opt.dSNC == 0 then
        	local outputs    = model:forward(inputs_gpu)
			confusion:batchAdd(outputs, targets_gpu)
		elseif opt.dSNC == 1 then
       		local features    = lastOutput(model, inputs_gpu)
        	local outputs     = dknn(data.xz, data.yz, features)
			confusion:batchAdd(outputs, targets_gpu)
		else
			print('**** error for opt.dSNC (0 or 1) ****')
		end
    end
    confusion:updateValids()
    err    = 1 - confusion.totalValid
    confusion:zero()
	return err
end

function reportErr(data, model, opt, confusion)
    local bestValid = math.huge
    local bestTest  = math.huge
    local bestTrain = math.huge
    local bestEpoch = math.huge
    local function report(t)
		local flag=0
        local err_e = evaluation('e', data, model, opt.batchSize, confusion)
        local err_v = evaluation('v', data, model, opt.batchSize, confusion)
        local err_r = evaluation('r', data, model, opt.batchSize, confusion)
        print('---------------Epoch: ' .. t .. ' of ' .. opt.max_epoch)
        print(string.format('Current Errors: test: %.4f | valid: %.4f | train: %.4f',
              err_e*100, err_v*100, err_r*100))
        if bestValid > err_v then
            bestValid = err_v
            bestTrain = err_r
            bestTest  = err_e
            bestEpoch = t
			flag=1
        end
        paths.mkdir(opt.save)
        testLogger:add{err_r*100, err_v*100, bestTest*100}
        testLogger:style{'-', '-', '-'}
        print(string.format('Optima achieved at epoch %d: test: %.4f, valid: %.4f',
              bestEpoch, bestTest*100, bestValid*100))
		return flag
    end
    return report
end
