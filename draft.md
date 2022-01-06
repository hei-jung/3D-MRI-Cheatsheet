```python
for param in model.parameters():
    param.requires_grad = False
    
n_inputs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 1),
    nn.LogSoftmax(dim=1)
)
```

```python
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

```python
def train(model, criterion, optimizer, train_loader, valid_loader):
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    valid_max_acc = 0
    history = []
    
    for epoch in range(60):
        train_loss = 0.0
        valid_loss = 0.0
        
        train_acc = 0.0
        valid_acc = 0.0
        
        model.train()
        
        for inputs, target in train_loader:
            inputs, target = Variable(inputs), Variable(target)
            
            optimizer.zero_grad()
            output = model(inputs)
            
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item() * inputs.size(0)
            
            print(f'Epoch: {epoch}')
```

```python
train(model, criterion, optimizer, train_loader, valid_loader)
```
