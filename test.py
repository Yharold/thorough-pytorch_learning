from torchvision import models
from torchinfo import summary

resnet18 = models.resnet18()

summary(resnet18, (1, 3, 224, 224))

print("")