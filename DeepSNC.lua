require 'metric.lua'
local DeepSNC, Parent = torch.class('nn.DeepSNC', 'nn.Module')

function DeepSNC:__init(z, zy, opt)
	Parent.__init(self)
	self.z 	= z
	self.gradz = torch.CudaTensor(z:size(1), z:size(2))
	self.zy = zy
	self.gamma = opt.gamma
	self.lambda= opt.lambda
	self.lambda2 = opt.lambda2
	self.Qz   = Deltaij(self.zy, self.zy)
	self.m  = z:size(1)
	self.d  = z:size(2)
	self.gradInput:cuda()
end

function DeepSNC:updateOutput(input, target)
	self.n 			= input:size(1)
	if self.lambda ~= 0 then
		self.dist		= spdist2(input, self.z)
		f1 = (self.dist-self.dist:min(2):expand(self.n, self.m) ):mul(-self.gamma*self.gamma):exp():apply(function(x) if x<1e-19*self.n then return 1e-19*self.n else return x end end)	
		self.Pij		= f1:cdiv(f1:sum(2):expand(self.n, self.m))
	else
		self.Pij 		= GetPij(input, self.z, self.gamma)
	end
	self.mDeltaij 	= Deltaij(target, self.zy)
	self.Pi         = torch.cmul(self.Pij, self.mDeltaij):sum(2):expand(self.n, self.m)
	self.mPij 		= torch.cdiv(self.Pij, self.Pi)
	if self.lambda ~=0 then 
		self.output		= torch.cmul(self.Pij, self.mDeltaij):sum(2):log():mul(-1):sum()+opt.lambda*torch.sum(self.dist) + opt.lambda2*torch.cmul(self.Qz, spdist2(self.z, self.z)):sum()
	else
		self.output     = torch.cmul(self.Pij, self.mDeltaij):sum(2):log():mul(-1):sum()
	end
	return self.output
end

function DeepSNC:updateGradInput(input,target) 
	self.mxQP = torch.cmul(self.mDeltaij, self.mPij):t()
	if self.lambda ~= 0 then
		self.gradInput:resize(input:size()):copy(  (torch.mm(self.z:t(), - self.Pij:t() + self.mxQP) * (-2) * self.gamma * self.gamma):t() + (input*self.m - torch.sum(self.z, 1):expand(self.n, self.d))*2*self.lambda)
	else
		self.gradInput:resize(input:size()):copy(  (torch.mm(self.z:t(), - self.Pij:t() + self.mxQP) * (-2) * self.gamma * self.gamma):t() )
	end
	return self.gradInput
end

function DeepSNC:accGradParameters(input, target)
	self.gradz:zero()
	self.mzQP = torch.cmul(self.mDeltaij - self.Pi, self.mPij)
	if self.lambda ~= 0 then
		self.gradz:add( (torch.mm(input:t(), self.mzQP) - torch.mm(self.z:t(),  vector2matrix(self.mzQP:t()*torch.ones(self.n, 1):cuda())  ) ):t()*(-2)*self.gamma*self.gamma + (self.z*self.n - torch.sum(input, 1):expand(self.m, self.d))*2*self.lambda  + (self.z:t()*self.Qz - torch.cmul(self.Qz:sum(1):expand(self.d, self.m),self.z:t()) ):t()*2*self.lambda2)
	else
		self.gradz:add( (torch.mm(input:t(), self.mzQP) - torch.mm(self.z:t(),  vector2matrix(self.mzQP:t()*torch.ones(self.n, 1):cuda())  ) ):t()*(-2)*self.gamma*self.gamma )
	end
end

function DeepSNC:forward(input, target)
	return self:updateOutput(input, target)
end

function DeepSNC:backward(input, target)
   self:accGradParameters(input, target)
   return self:updateGradInput(input, target)
end

function DeepSNC:parameters()
	return {self.z}, {self.gradz}
end
