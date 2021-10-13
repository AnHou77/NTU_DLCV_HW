import torch
device = 'cuda:0'

N = 64
D_in = 1000
H = 100
D_out = 10

x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)
)

learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(50):
    y_pred = model(x)
    
    loss = torch.nn.functional.mse_loss(y_pred,y)

    loss.backward()

    print(loss)

    optimizer.step()
    optimizer.zero_grad()