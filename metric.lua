--  ========================
--  calculate euclident dist between two matrix
--  ========================
function spdist2(x, y)
-- x : n * d
-- y : m * d
-- return res : n * m
local n = x:size(1)
local m = y:size(1)
local dist1 = torch.cmul(x, x):sum(2):expand(n, m)
local dist2 = torch.cmul(y, y):sum(2):t():expand(n,m)
local dist3 = torch.mm(x, y:t()):mul(-2)
return dist3:add(dist1):add(dist2)
end 

function GetPij(x, z, gamma)
-- x : n * d
-- z : m * d  
-- return res : n * m  probability x_i belongs to z_j
local n = x:size(1)
local m = z:size(1)
local eps = 1e-19*n
local f1 = (spdist2(x, z)-spdist2(x, z):min(2):expand(n,m) ):mul(-gamma*gamma):exp():apply(function(x) if x<eps then return eps else return x end end)
return f1:cdiv(f1:sum(2):expand(n,m))

end

function Deltaij(xy, zy)
-- xy : n
-- zy : m
local n = xy:size(1)
local m = zy:size(1)
return xy:reshape(n,1):expand(n, m):eq(zy:reshape(m,1):t():expand(n, m))
end

function vector2matrix(vector)
-- return a square matrix -- diagnal to be vector
-- since there is no torch.diag for cuda()
-- vector : n -by- 1
local n = vector:size(1)
local v = vector:expand(n,n)
if v:type() == 'torch.CudaTensor' then
	return torch.cmul(v, torch.eye(n):cuda())
else
	return torch.cmul(v, torch.eye(n))
end
end
