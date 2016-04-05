require 'metric.lua'

function dknn(xr, yr, xe)
	-- 	xr : m -by- d
	-- 	yr : m 
	-- 	xe : n -by- d
	-- 	return : n  lable
	local n = xe:size(1)
	local m = xr:size(1)
	local dist = spdist2(xe, xr)
	local _, rank = torch.min(dist, 2)
	local label = torch.CudaTensor():resize(n,m):copy(yr:resize(1, m):expand(n, m))
	yr:resize(m, 1)
	return label:gather(2, rank):resize(n)
end


function dknn_eval(y, ye)
	--	 y : m true lable
	--	 ye: m predicted label
	--   return error rate
	local m = y:size(1)
	local res = (m - (y:resize(m,1)-ye:resize(m, 1)):eq(0):sum() )/m
	y:resize(m)
	ye:resize(m)
	return res
end
