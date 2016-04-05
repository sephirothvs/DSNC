require 'nn'
require 'cunn'
require 'cudnn'
model = nn.Sequential()

model:add(cudnn.SpatialConvolution(1, 20, 5, 5))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

model:add(cudnn.SpatialConvolution(20, 50, 5, 5))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

model:add(nn.Reshape(50*4*4))
model:add(nn.Linear(50*4*4, opt.dim))
-- model:add(nn.ReLU())
-- model:add(nn.Linear(500, 200))
-- cudnn.convert(model, cudnn)
if opt.dSNC == 0 then
	--model:add(cudnn.ReLU())
	model:add(nn.Linear(opt.dim, opt.nClass))
end

return model 
