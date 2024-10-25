"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )

  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

class effnet_model(torch.nn.Module):
    # Initialize the parameters
    def __init__(self, model_name,effnet, weights):
        super(effnet_model, self).__init__()
        self.weights = weights
        self.model_name = model_name
        self.effnet = effnet(weights=self.weights)
        self.effnet.classifier = torch.nn.Identity()
        self.drp1 = torch.nn.Dropout(0.2)
        self.f1 = torch.nn.Flatten()
        self.dense3 = torch.nn.ReLU(torch.nn.Linear(64,128))
        self.drp3 = torch.nn.Dropout(0.5)
        self.dense4 =  torch.nn.ReLU(torch.nn.Linear(128,32))
        self.drp4 = torch.nn.Dropout(0.5)
        self.output_condition =  torch.nn.Linear(1280,3)

    def forward(self,inputs):
        x = self.effnet(inputs)
        x = self.drp1(x)
        x = self.f1(x)
        x = self.dense3(x)
        x = self.drp3(x)
        x = self.dense4(x)
        x = self.drp4(x)
        output = self.output_condition(x)
        return output

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):  

        """
        Args:
            inputs: A float tensor of size [batch_size, num_classes].
            targets: A long tensor of size [batch_size].
        """
        # Compute the cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        # Calculate the focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss