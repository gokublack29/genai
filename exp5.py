!pip install torch torchvision matplotlib
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()
def segment_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.Resize(520), T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[1].imshow(mask, cmap='jet')
    ax[1].set_title("Segmented Image")
    plt.show()
image_path = "C:\\Users\\lasha\\Downloads\\kids playing.jpg"
segment_image(image_path)

!pip install torch supervision matplotlib
import torch
from PIL import Image
import matplotlib.pyplot as plt
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
def detect_objects(image_path):
    img = Image.open(image_path)
    results = model(img)
    results.show()
    print(results.pandas().xyxy[0])
image_path = "C:\\Users\\lasha\\Downloads\\kids playing.jpg"
detect_objects(image_path)
