import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 10**7  
try:
    while True:
        a = torch.rand(n, device=device)
        b = torch.rand(n, device=device)
        result = a * b + torch.sqrt(a) - torch.log(b + 1e-6)
        total = result.sum()

except KeyboardInterrupt:
    print("end”)
