import math
import os
from PIL import Image
import numpy as np
import torch
from ImageUtils import *
from SatelliteImageGNN import SatelliteImageGNN
from matplotlib import pyplot as plt
import sys
import time
from matplotlib.patches import Rectangle

ML = 3.3420E-04
AL = 0.10000
K1 = 774.8853
K2 = 1321.0789
IMAGE_SIZE = 6000
FOLDER_NAME = 'images'

def dnToTemp(DN):
    return K2 / math.log(K1 / (ML * DN + AL) + 1)

def getInputOutput(img):
    firstTopLeft = getFirstPixelTopLeft(img)
    firstLeftTop = getFirstPixelLeftTop(img)
    slope = (firstLeftTop[1] - firstTopLeft[1]) / (firstLeftTop[0] - firstTopLeft[0])
    degrees = -math.degrees(math.atan(slope))
    img = img.rotate(degrees)
    img = img.crop((getFirstCol(img), getFirstRow(img), getLastCol(img) + 1, getLastRow(img) + 1))
    if img.height < IMAGE_SIZE or img.width < IMAGE_SIZE:
        return None
    img = img.crop((0, 0, IMAGE_SIZE, IMAGE_SIZE))
    output = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            DN = img.getpixel((j, i))
            if DN == 0:
                output[i][j] = 0
            else:
                output[i][j] = K2 / math.log(K1 / (ML * DN + AL) + 1)
    output = (output - np.min(output)) / (np.max(output) - np.min(output))
    input = np.zeros((IMAGE_SIZE // 3, IMAGE_SIZE // 3))
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            input[i // 3][j // 3] += img.getpixel((j, i)) / 9 # output[i][j] / 9
    return input, output

def load_images():
    inputs = torch.tensor(np.stack(np.load("input.npy"), axis=0), dtype=torch.float32)
    outputs = torch.tensor(np.stack(np.load("output.npy"), axis=0), dtype=torch.float32)
    return inputs, outputs

def saveFolder(folderPath):
    inputs = []
    outputs = []
    for filePath in os.listdir(folderPath):
        inputOutput = getInputOutput(Image.open(os.path.join(folderPath, filePath)))
        if inputOutput is not None:
            input, output = inputOutput
            inputs.append(input)
            outputs.append(output)
    np.save('input.npy', np.array(inputs))
    np.save('output.npy', np.array(outputs))

def mainTrain():
    # Load your data
    inputs, outputs = load_images()

    # Create model with resource limits (6GB RAM, 2 CPU cores)
    model = SatelliteImageGNN(
        scale_factor=3,
        max_memory_gb=6,
        num_cores=4
    )
    print("Model creat, data read")
    # Train model
    try:
        model.train_model(inputs, outputs, epochs=5, batch_size=1)
        torch.save(model.state_dict(), 'satellite_model.pth')
        print("Training completed successfully!")
    except MemoryError as e:
        print(f"Training failed due to memory constraints: {e}")
        print("Try reducing batch_size or using smaller chunks of data")
    except Exception as e:
        print(f"An error occurred: {e}")

def compare():
    ml = load_trained_model()
    input_image = np.load("pozainput.npy")
    print(input_image.shape)
    with torch.no_grad():
        graph_data = ml.create_graph(input_image)
        prediction = ml(graph_data)
        output = prediction.cpu().numpy().squeeze()

    # Create visualization
    print("Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot input image
    im1 = ax1.imshow(input_image, cmap='viridis')
    ax1.set_title('Input Image (2000x2000)')
    plt.colorbar(im1, ax=ax1)

    # Plot super-resolved output
    im2 = ax2.imshow(output, cmap='viridis')
    ax2.set_title('Super-resolved (6000x6000)')
    plt.colorbar(im2, ax=ax2)
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_trained_model(model_path='BigModel.pth', scale_factor=3, hidden_channels=32):
    model = SatelliteImageGNN(
        scale_factor=scale_factor,
        hidden_channels=hidden_channels
    )
    try:
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {model_path}")
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def processImage(filename):
    arr = np.array(Image.open(filename))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j] = dnToTemp(arr[i][j])
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr

def mainConsole():
    ml = load_trained_model()
    input_image = processImage('test.TIF' if len(sys.argv) == 1 else sys.argv[1])
    print(input_image.shape)
    with torch.no_grad():
        graph_data = ml.create_graph(input_image)
        prediction = ml(graph_data)
        output = prediction.cpu().numpy().squeeze()
    print("Creating visualization...")
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:])
    x_start, y_start = 1250, 1500
    width, height = 100, 100
    out_x = x_start * 3
    out_y = y_start * 3
    out_w = width * 3
    out_h = height * 3
    im1 = ax1.imshow(input_image, cmap='viridis')
    ax1.set_title('Input Image (2000x2000)')
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(output, cmap='viridis')
    ax2.set_title('Super-resolved (6000x6000)')
    plt.colorbar(im2, ax=ax2)
    rect1 = Rectangle((x_start, y_start), width, height,
                    fill=False, color='red', linewidth=2)
    rect2 = Rectangle((out_x, out_y), out_w, out_h,
                    fill=False, color='red', linewidth=2)
    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    ax3.imshow(input_image[y_start:y_start+height, x_start:x_start+width], cmap='viridis')
    ax3.set_title('Input Image (Zoomed)')
    ax4.imshow(output[out_y:out_y+out_h, out_x:out_x+out_w], cmap='viridis')
    ax4.set_title('Super-resolved (Zoomed)')
    plt.tight_layout()
    plt.savefig('comparison_plot_with_zoom.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    mainConsole()