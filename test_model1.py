import torch
import torchvision.models as models
from torchvision import transforms
from vgg16 import VGG16
from PIL import Image

num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(num_classes=num_classes).to(device)

# Use line below to load your .pth file for trained weights
model.load_state_dict(torch.load('best_vgg16.pth', map_location=device))

### Prep image to see model work
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## Load image you would like to test on
image = Image.open("Eosinophil.jpg")
input_tensor = transform(image).unsqueeze(0)
input_tensor = input_tensor.to(device)
model.eval()
# Perform inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output,dim=1).item()

#Prediction value 0 - Basophil, 1 - Eosinophil, 2 - Lymphocyte, 3 - Monocyte, 4-Neutrophil 
print(f"Predicted Class: {prediction}")