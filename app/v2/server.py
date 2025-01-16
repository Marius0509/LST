import os
import math
import numpy as np
from PIL import Image
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import time

def getTopLeft(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] != 0:
                return i, j

def getTopRight(arr):
    for j in range(arr.shape[1] - 1, -1, -1):
        for i in range(arr.shape[0]):
            if arr[i][j] != 0:
                return i, j

def getBottomRight(arr):
    for i in range(arr.shape[0] - 1, -1, -1):
        for j in range(arr.shape[1] - 1, -1, -1):
            if arr[i][j] != 0:
                return i, j

def getBottomLeft(arr):
    for j in range(arr.shape[1]):
        for i in range(arr.shape[0] - 1, -1, -1):
            if arr[i][j] != 0:
                return i, j

ML = 0.0003342
AL = 0.1
K1 = 774.8853
K2 = 1321.0789

def dnToTemp(dn):
    return K2 / math.log(K1 / (ML * dn + AL) + 1)

def tempToDn(temp):
    return round((K1 / (math.exp(K2 / temp) - 1) - AL) / ML)

MIN_TEMP = dnToTemp(0)
MAX_TEMP = dnToTemp(65535)

def dnToNormalizedTempArr(arr):
    newArr = np.empty((arr.shape[0], arr.shape[1]), np.float64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            newArr[i][j] = dnToTemp(arr[i][j])
    newArr = (newArr - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
    return newArr

def normalizedTempToDnArr(arr):
    arr = arr * (MAX_TEMP - MIN_TEMP) + MIN_TEMP
    newArr = np.empty((arr.shape[0], arr.shape[1]), np.uint16)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            newArr[i][j] = tempToDn(arr[i][j])
    return newArr

def processImages():
    files = os.listdir(os.path.join('data', 'images'))
    for counter, file in enumerate(files):
        arr = np.array(Image.open(os.path.join('data', 'images', file)))
        topLeft = getTopLeft(arr)
        topRight = getTopRight(arr)
        bottomRight = getBottomRight(arr)
        bottomLeft = getBottomLeft(arr)
        if bottomLeft[0] - topRight[0] + 1 < 3000 or bottomRight[1] - topLeft[1] + 1 < 3000:
            print(f'file {file} is too small')
        else:
            filename = os.path.join('data', 'npy', '.'.join(file.split('.')[:-1] + ['npy']))
            arr = dnToNormalizedTempArr(arr[topRight[0]:topRight[0] + 3000, topLeft[1]:topLeft[1] + 3000])
            np.save(filename, arr)
        print(f'{counter + 1}/{len(files)} images processed')

def processData():
    inputs = []
    outputs = []
    files = os.listdir(os.path.join('data', 'npy'))
    for counter, file in enumerate(files):
        arr = np.load(os.path.join('data', 'npy', file))
        for i in range(40):
            for j in range(40):
                input = np.zeros((25, 25))
                for ii in range(75):
                    for jj in range(75):
                        input[ii // 3][jj // 3] += arr[i * 75 + ii][j * 75 + jj] / 9
                inputs.append(input)
                output = np.empty((3, 3))
                for ii in range(3):
                    for jj in range(3):
                        output[ii][jj] = arr[i * 75 + 36 + ii][j * 75 + 36 + jj]
                outputs.append(output)
        print(f'{counter + 1}/{len(files)} files processed')
    np.save(os.path.join('data', 'inputs.npy'), np.array(inputs))
    np.save(os.path.join('data', 'outputs.npy'), np.array(outputs))

def trainModel(inputs, outputs):
    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(42)
    model = keras.models.Sequential([
        keras.layers.Input(shape=(25, 25, 1)),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(9, activation='sigmoid'),
        keras.layers.Reshape((3, 3))
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    trainInputs, validationInputs, trainOutputs, validationOutputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    earlyStopping = keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', verbose=1)
    modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath='model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    history = model.fit(
        trainInputs, trainOutputs,
        validation_data=(validationInputs, validationOutputs),
        batch_size=128,
        epochs=1_000_000,
        callbacks=[earlyStopping, modelCheckpoint]
    )
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], marker='o')
    plt.grid(True)
    plt.savefig('loss.png', dpi=300)
    plt.close()

def zoomIn(input, model):
    input = dnToNormalizedTempArr(input)
    output = np.empty((input.shape[0] * 3 - 72, input.shape[1] * 3 - 72))
    for i in range(input.shape[0] - 24):
        patches = np.empty((input.shape[1] - 24, 25, 25, 1))
        for j in range(input.shape[1] - 24):
            patches[j] = input[i:i + 25, j:j + 25].reshape(25, 25, 1)
        prediction = model.predict(patches, verbose=0)
        for j in range(input.shape[1] - 24):
            for ii in range(3):
                for jj in range(3):
                    output[i * 3 + ii][j * 3 + jj] = prediction[j][ii][jj]
    return normalizedTempToDnArr(output)

def zoomOut(input):
    height, width = input.shape
    if height % 3 != 0 or width % 3 != 0:
        raise Exception('can\'t zoom out arr')
    output = np.zeros((height // 3, width // 3), dtype = np.uint16)
    for i in range(height // 3):
        for j in range(width // 3):
            sum = 0
            for ii in range(3):
                for jj in range(3):
                    sum += dnToTemp(input[i * 3 + ii][j * 3 + jj])
            output[i][j] = tempToDn(sum / 9)
    return output

def comparisonPlot(fileIn, fileOut):
    input = np.array(Image.open(fileIn)).astype(np.float64)
    output = np.array(Image.open(fileOut)).astype(np.float64)
    input = input[12:input.shape[0] - 12, 12:input.shape[1]-12]
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            input[i][j] = dnToTemp(input[i][j]) - 273.15
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] = dnToTemp(output[i][j]) - 273.15
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:])
    width, height = 10, 10
    x_start, y_start = (input.shape[0] - width) // 2, (input.shape[1] - height) // 2
    out_x = x_start * 3
    out_y = y_start * 3
    out_w = width * 3
    out_h = height * 3
    im1 = ax1.imshow(input, cmap='viridis')
    ax1.set_title(f'Input Image ({input.shape[0]}x{input.shape[1]})')
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(output, cmap='viridis')
    ax2.set_title(f'Super-resolved ({output.shape[0]}x{output.shape[1]})')
    plt.colorbar(im2, ax=ax2)
    rect1 = Rectangle((x_start, y_start), width, height, fill=False, color='red', linewidth=2)
    rect2 = Rectangle((out_x, out_y), out_w, out_h, fill=False, color='red', linewidth=2)
    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    ax3.imshow(input[y_start:y_start+height, x_start:x_start+width], cmap='viridis')
    ax3.set_title('Input Image (Zoomed)')
    ax4.imshow(output[out_y:out_y+out_h, out_x:out_x+out_w], cmap='viridis')
    ax4.set_title('Super-resolved (Zoomed)')
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')

def getMAE(arr1, arr2):
    return np.mean(np.abs(arr1 - arr2))

def getMSE(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)

def getR2(arr1, arr2):
    ss_res = np.sum((arr1 - arr2) ** 2)
    ss_tot = np.sum((arr1 - np.mean(arr2)) ** 2)
    return  1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')

def computeMetrics(model):
    metrics = []
    files = os.listdir(os.path.join('data', f'images'))
    for count, filename in enumerate(files):
        arr = np.array(Image.open(os.path.join('data', 'images', filename)))
        x, y = getTopRight(arr)[0], getTopLeft(arr)[1]
        arr = arr[x:x + 3072, y:y + 3072].astype(np.float64)
        groundTruth = arr[36:3036, 36:3036]
        output = zoomIn(zoomOut(arr), model)
        mae = getMAE(groundTruth, output)
        mse = getMSE(groundTruth, output)
        r2 = getR2(groundTruth, output)
        metrics.append(f'{filename} {mae} {mse} {r2}')
        print(f'{count + 1}/{len(files)} files done')
    with open(f'metrics.txt', 'w') as file:
        for x in metrics:
            file.write(f'{x}\n')

def printAverageMetrics():
    maeSum, mseSum, r2Sum, count = 0, 0, 0, 0
    with open('metrics.txt', 'r') as file:
        for line in file.readlines():
            _, mae, mse, r2 = line.split()
            maeSum += float(mae)
            mseSum += float(mse)
            r2Sum += float(r2)
            count += 1
    print(f'mae: {round(maeSum / count, 2)}')
    print(f'mse: {round(mseSum / count, 2)}')
    print(f'r2: {round(r2Sum / count, 2)}')


app = Flask(__name__)
CORS(app)
model = None

@app.route('/upload', methods=['POST'])
def upload_file():
    input = np.array(Image.open(io.BytesIO(request.files['file'].read())))
    inputShape = input.shape
    print(f'received {inputShape} image')
    if inputShape[0] < 25 or inputShape[1] < 25:
        return jsonify({'error': 'image is too small'}), 400
    start = time.time()
    output = zoomIn(input, model)
    end = time.time()
    outputShape = output.shape
    print(f'sent {outputShape} image')
    print(f'{inputShape} -> {outputShape} in {round(end - start, 3)} seconds')
    output = Image.fromarray(output)
    img_io = io.BytesIO()
    output.save(img_io, format='TIFF')
    img_io.seek(0)
    return send_file(
        img_io,
        mimetype='image/tiff'
    )

if __name__ == '__main__':
    # processImages()
    # processData()
    # inputs = np.load(os.path.join('data', 'inputs.npy'))
    # outputs = np.load(os.path.join('data', 'outputs.npy'))
    # trainModel(inputs, outputs)
    # model = keras.models.load_model('model.keras')
    # comparisonPlot('input.TIF', 'output.TIF')
    # computeMetrics(model)
    # printAverageMetrics()
    
    model = keras.models.load_model('model.keras')
    app.run(debug=True, host='127.0.0.1', port=5000)