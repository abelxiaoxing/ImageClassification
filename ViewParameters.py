from torchstat import stat
from model.cyclemlp import CycleMLP_B1
model = CycleMLP_B1(in_chans=3,num_classes=2)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameters: %.2fk" % (total/1e3))
