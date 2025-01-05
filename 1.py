import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#qyx
epochs= 10000 #迭代次数
h = 100#网格数
N = 1000#内点数
N1=100#边界点数


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(888888)#设置固定的随机数种子，方便结果复现

#对方程及边界条件参数化
def interior(n=N):
    x = torch.rand(n,1)
    t = torch.rand(n, 1)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True),t.requires_grad_(True),cond
#应数
def down(n=N1):
    x = torch.rand(n,1)
    t = torch.zeros_like(x)
    cond = (1/torch.cosh(x+5))**2
    return x.requires_grad_(True),t.requires_grad_(True),cond
def left(n=N1):
    t = torch.rand(n, 1)
    x = (-10)*torch.ones_like(t)
    cond = torch.rand(n,1)
    return x.requires_grad_(True),t.requires_grad_(True),cond
def right(n=N1):
    t = torch.rand(n, 1)
    x = (10) * torch.ones_like(t)
    a,b,c = left(n)
    cond = c
    return x.requires_grad_(True), t.requires_grad_(True), cond
def px_left(n=N1):
    t = torch.rand(n, 1)
    x = (-10) * torch.ones_like(t)
    cond = torch.rand(n, 1)
    return x.requires_grad_(True), t.requires_grad_(True), cond
def px_right(n=N1):
    t = torch.rand(n, 1)
    x = (10) * torch.ones_like(t)
    a, b, c = px_left(n)
    cond = c
    return x.requires_grad_(True), t.requires_grad_(True), cond

#2021
#定义神经网络
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32,1)
        )
    def forward(self,x):
        return self.net(x)
#损失函数
loss = torch.nn.MSELoss()
#递归算导数
def gradients(u,x,order = 1):
    if order==1:
        return torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True,only_inputs=True,)[0]
    else :
        return gradients(gradients(u,x),x,order=order-1)
#7个方程及边界条件的loss
def l_interior(u):
    x,t,cond = interior()
    uxt = u(torch.cat([x,t],dim=1))
    return loss(gradients(uxt,t,1)+6*uxt*gradients(uxt,x,1)+gradients(uxt,x,3),cond)
#0514
def l_down(u):
    x, t, cond = down()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(uxt,cond)
def l_left(u):
    x, t, cond = left()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(uxt, cond)
def l_right(u):
    x, t, cond = right()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(uxt, cond)
def l_px_left(u):
    x, t, cond = px_left()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt,x,1),cond)
def l_px_right(u):
    x, t, cond = px_right()
    uxt = u(torch.cat([x, t], dim=1))
    return loss(gradients(uxt,x,1),cond)
#0115
u = MLP()
opt = torch.optim.Adam(params=u.parameters())
#使用Adam优化器，迭代一万次优化
for i in range(epochs):
    opt.zero_grad()
    l = l_interior(u)+l_down(u)+l_left(u)+l_right(u)+l_px_left(u)+l_px_right(u)
    l.backward()
    opt.step()
    if i%100==0:
        print(i)

#绘图
xc=torch.linspace(0,1,h)
xm,ym = torch.meshgrid(xc,xc)
xx = xm.reshape(-1,1)
yy = ym.reshape(-1,1)
xy = torch.cat([xx,yy],dim=1)
u_pred = u(xy)
u_pred_fig = u_pred.reshape(h,h)

fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot_surface(xm.detach().numpy(),ym.detach().numpy(),u_pred_fig.detach().numpy())
ax.text2D(0.5,0.9,"PINN",transform=ax.transAxes)
plt.show()
fig.savefig("PINN solve.png")

