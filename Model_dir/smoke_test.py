from Model_Classes import GPT
from HyperParam_Classes import Config
from torchinfo import summary
import torch

device="cuda" if torch.cuda.is_available() else "cpu"

model=GPT(Config()).to(device)
# model=torch.compile(model).to(device)

model(torch.randint(0,32786,(1,1024)).to(device))
print("Input passed through the model successfully!")

print(summary(model,input_size=(1,1024),dtypes=[torch.long],device=device))