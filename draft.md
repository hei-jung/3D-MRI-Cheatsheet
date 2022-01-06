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
