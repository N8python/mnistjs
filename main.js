import { init } from "@thi.ng/simd";
import { MemPool } from "@thi.ng/malloc";

import fs from 'fs';
// Initialize SIMD once at the start of your application
const simd = init(
    new WebAssembly.Memory({
        initial: Math.ceil((1024 * 1024 * 32) / 0x10000), // Allocate 4MB of memory
    })
);

// Create a memory pool for managing buffers
const pool = new MemPool({
    buf: simd.memory.buffer,
    align: 16, // Align to 16 bytes for SIMD operations
});

const utilBufferCache = {}
const BATCH_SIZE = 128;
const LEARNING_RATE = 1;

class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new Float32Array(this.rows * this.cols);
    }

    fromArray(arr) {
        /* const alen = arr.length;
         for (let i = 0; i < alen; i++) {
             this.data[i] = arr[i];
         }*/
        this.data.set(arr);
        return this;
    }
    copy() {
        let m = new Matrix(this.rows, this.cols);
        m.data.set(this.data);
        return m;
    }
    add(n, c = 1) {
        let m = this.copy();
        const dLen = this.data.length;
        const mData = m.data;
        if (n instanceof Matrix) {
            const nData = n.data;
            for (let i = 0; i < dLen; i++) {
                mData[i] += nData[i] * c;
            }
        } else {
            for (let i = 0; i < dLen; i++) {
                mData[i] += n * c;
            }
        }
        return m;
    }
    addInPlace(n, c = 1) {
        const dLen = this.data.length;
        const data = this.data;
        if (n instanceof Matrix) {
            const nData = n.data;
            for (let i = 0; i < dLen; i++) {
                data[i] += nData[i] * c;
            }
        } else {
            for (let i = 0; i < dLen; i++) {
                data[i] += n * c;
            }
        }
    }
    addAcrossBatch(n) {
        const rows = this.rows;
        const cols = this.cols;
        let m = this.copy();
        const mData = m.data;
        const nData = n.data;
        for (let i = 0; i < cols; i++) {
            for (let j = 0; j < rows; j++) {
                mData[j * cols + i] += nData[j];
            }
        }
        return m;
    }

    multiply(n) {
        let m = this.copy();
        const mData = m.data;
        if (n instanceof Matrix) {
            const nData = n.data;
            const dLen = this.data.length;
            for (let i = 0; i < dLen; i++) {
                mData[i] *= nData[i];
            }
        } else {
            const dLen = this.data.length;
            for (let i = 0; i < dLen; i++) {
                mData[i] *= n;
            }
        }
        return m;
    }
    transpose() {
        let result = new Matrix(this.cols, this.rows);
        const rows = this.rows;
        const cols = this.cols;
        const mData = this.data;
        const resData = result.data;

        let i = 0;

        // Handle the bulk of the data in blocks of 4
        for (; i <= rows - 4; i += 4) {
            const rowOffset0 = i * cols;
            const rowOffset1 = (i + 1) * cols;
            const rowOffset2 = (i + 2) * cols;
            const rowOffset3 = (i + 3) * cols;

            for (let j = 0; j < cols; j++) {
                const colIndex = j * rows;
                resData[colIndex + i] = mData[rowOffset0 + j];
                resData[colIndex + i + 1] = mData[rowOffset1 + j];
                resData[colIndex + i + 2] = mData[rowOffset2 + j];
                resData[colIndex + i + 3] = mData[rowOffset3 + j];
            }
        }

        // Handle any remaining rows that weren't covered by the unrolled loop
        for (; i < rows; i++) {
            const rowOffset = i * cols;
            for (let j = 0; j < cols; j++) {
                resData[j * rows + i] = mData[rowOffset + j];
            }
        }

        return result;
    }
    applyElementWise(fn) {
        let result = new Matrix(this.rows, this.cols);
        const resultData = result.data;
        const myData = this.data;
        const myDataLen = myData.length;
        for (let i = 0; i < myDataLen; i++) {
            resultData[i] = fn(myData[i]);
        }
        return result;
    }
    meanOverBatch() {
        let result = new Matrix(this.rows, 1);
        const resultData = result.data;
        const myData = this.data;
        const nRows = this.rows;
        const nCols = this.cols;
        for (let i = 0; i < nRows; i++) {
            let sum = 0;
            const iNcols = i * nCols;
            /*for (let j = 0; j < nCols; j++) {
                sum += myData[iNcols + j];
            }*/
            let j = 0;
            for (; j <= nCols - 4; j += 4) {
                sum += myData[iNcols + j];
                sum += myData[iNcols + j + 1];
                sum += myData[iNcols + j + 2];
                sum += myData[iNcols + j + 3];
            }
            for (; j < nCols; j++) {
                sum += myData[iNcols + j];
            }
            resultData[i] = sum / nCols;
        }
        return result;
    }

    static gemm(a, b) {
        if (a.cols !== b.rows) {
            console.error('Columns of A must match rows of B');
            return undefined;
        }

        let result = new Matrix(a.rows, b.cols);
        const resRows = a.rows;
        const resCols = b.cols;
        const aCols = a.cols;

        // Ensure aCols is a multiple of 4, pad if necessary
        const paddedACols = Math.ceil(aCols / 4) * 4;

        // Calculate total memory needed

        // Allocate memory for matrices A, B (transposed), and dot product buffer
        const key = `${paddedACols}_${resRows}_${resCols}`;
        if (!utilBufferCache[key]) {
            utilBufferCache[key] = {
                bufferA: pool.callocAs("f32", paddedACols * resRows),
                bufferBT: pool.callocAs("f32", paddedACols * resCols),
                dotProductBuffer: pool.callocAs("f32", paddedACols)
            };
        }
        const { bufferA, bufferBT, dotProductBuffer } = utilBufferCache[key];

        const aData = a.data;
        const bData = b.data;
        // Copy matrix A to SIMD memory
        for (let i = 0; i < resRows; i++) {
            const iACols = i * aCols;
            bufferA.set(aData.subarray(iACols, iACols + aCols), i * paddedACols);

            // Pad with zeros if necessary
            const iPadding = i * paddedACols;
            for (let k = aCols; k < paddedACols; k++) {
                bufferA[iPadding + k] = 0;
            }
        }

        // Copy matrix B to SIMD memory (transposed)
        for (let j = 0; j < resCols; j++) {
            const jpaddedACols = j * paddedACols;
            for (let k = 0; k < aCols; k++) {
                bufferBT[jpaddedACols + k] = bData[k * resCols + j];
            }
            // Pad with zeros if necessary
            for (let k = aCols; k < paddedACols; k++) {
                bufferBT[jpaddedACols + k] = 0;
            }
        }
        const resultData = result.data;
        const dotProductBufferPtr = dotProductBuffer.byteOffset;
        const bufferAPtr = bufferA.byteOffset;
        const bufferBTPtr = bufferBT.byteOffset;
        const paddedAColsTimes4 = paddedACols * 4;
        const paddedAColsDiv4 = paddedACols / 4;
        for (let i = 0; i < resRows; i++) {
            const ipaddedAColsTimes4 = i * paddedAColsTimes4;
            const iResCols = i * resCols;
            for (let j = 0; j < resCols; j++) {
                // Compute dot product using SIMD
                simd.dot4_f32_aos(
                    dotProductBufferPtr,
                    bufferAPtr + ipaddedAColsTimes4,
                    bufferBTPtr + j * paddedAColsTimes4,
                    paddedAColsDiv4,
                    1, // output stride
                    4, // A stride (move to next vec4 in row)
                    4 // BT stride (move to next vec4 in row of transposed B)
                );
                let dotProduct = simd.sum4_f32(
                    dotProductBufferPtr,
                    paddedAColsDiv4,
                    4
                );
                resultData[iResCols + j] = dotProduct;
            }
        }

        return result;
    }
    static randomize(rows, cols, min = -1, max = 1) {
        let m = new Matrix(rows, cols);
        for (let i = 0; i < m.data.length; i++) {
            m.data[i] = Math.random() * (max - min) + min;
        }
        return m;
    }
    setColumn(columnIndex, values) {
        if (columnIndex < 0 || columnIndex >= this.cols) {
            throw new Error("Column index out of bounds");
        }
        if (values.length !== this.rows) {
            throw new Error("Values array length must match the number of rows");
        }
        const cols = this.cols;
        const rows = this.rows;
        const data = this.data;

        for (let i = 0; i < rows; i++) {
            data[i * cols + columnIndex] = values[i];
        }
    }

    getColumn(columnIndex) {
        if (columnIndex < 0 || columnIndex >= this.cols) {
            throw new Error("Column index out of bounds");
        }
        let column = new Float32Array(this.rows);
        for (let i = 0; i < this.rows; i++) {
            column[i] = this.data[i * this.cols + columnIndex];
        }
        return column;
    }
}
class Activation {
    constructor(fn, dfn) {
        this.fn = fn;
        this.dfn = dfn;
    }
}

const sigmoid = new Activation(
    x => 1 / (1 + Math.exp(-x)),
    y => y * (1 - y)
);

const tanh = new Activation(
    x => Math.tanh(x),
    y => 1 - y * y
);

class LinearLayer {
    constructor(input, output, activation) {
        this.weights = Matrix.randomize(output, input, -1 / Math.sqrt(input), 1 / Math.sqrt(input));
        this.bias = new Matrix(output, 1);
        this.activation = activation;
        this.inputShape = input;
        this.outputShape = output;
    }

    forward(input) {
        this.input = input;
        this.activated = Matrix.gemm(this.weights, input).addAcrossBatch(this.bias).applyElementWise(this.activation.fn);
        return this.activated;
    }

    backward(outputError, learningRate = 0.1) {
        const dActivated = outputError.multiply(this.activated.applyElementWise(this.activation.dfn));
        const dWeights = Matrix.gemm(dActivated, this.input.transpose());
        const dBias = dActivated.meanOverBatch();
        this.weights.addInPlace(dWeights, -learningRate);
        this.bias.addInPlace(dBias, -learningRate);
        return Matrix.gemm(this.weights.transpose(), dActivated);
    }
}

class MLP {
    constructor(layers) {
        this.layers = [];
        for (let i = 0; i < layers.length - 1; i++) {
            this.layers.push(new LinearLayer(layers[i], layers[i + 1], i === layers.length - 2 ? sigmoid : tanh));
        }
    }

    forward(input) {
        let output = input;
        for (let layer of this.layers) {
            output = layer.forward(output);
        }
        return output;
    }

    backward(error, learningRate = 0.01) {
        for (let i = this.layers.length - 1; i >= 0; i--) {
            error = this.layers[i].backward(error, learningRate);
        }
    }
}

function loadMNISTData(dataPath, labelsPath, numSamples) {
    const dataBuffer = fs.readFileSync(dataPath);
    const labelsBuffer = fs.readFileSync(labelsPath);

    const dataArray = new Uint8Array(dataBuffer);
    const labelsArray = new Uint8Array(labelsBuffer);

    const dataset = [];

    for (let i = 0; i < numSamples; i++) {
        const input = Array.from(dataArray.slice(i * 784, (i + 1) * 784)).map(x => x / 255);
        const output = new Array(10).fill(0);
        output[labelsArray[i]] = 1;

        dataset.push({ input, output });
    }

    return dataset;
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}
const trainingSet = loadMNISTData('mnist_train_data.bin', 'mnist_train_labels.bin', 60000);
const testSet = loadMNISTData('mnist_test_data.bin', 'mnist_test_labels.bin', 10000);

function createProgressBar(total, length = 30) {
    return function(current, max, epoch, loss, accuracy) {
        const filled = Math.round(length * current / max);
        const bar = '█'.repeat(filled) + '-'.repeat(length - filled);
        const percent = Math.round(100 * current / max);
        process.stdout.write(`\rEpoch ${epoch}: [${bar}] ${percent}% | ${current}/${max} iters, Loss: ${loss.toFixed(3)}, Accuracy: ${accuracy.toFixed(3)}`);
    };
}

function train(network, epochs = 10, learningRate = 0.001, batchSize = 128) {
    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalError = 0;
        let accuracy = 0;
        // Shuffle the training set
        shuffleArray(trainingSet);

        const updateProgressBar = createProgressBar(trainingSet.length);
        let batchIdx = 0;
        const totalBatches = Math.ceil(trainingSet.length / batchSize);
        const startTime = Date.now();
        for (let i = 0; i < trainingSet.length; i += batchSize) {
            const batch = trainingSet.slice(i, i + batchSize);
            const inputs = new Matrix(network.layers[0].inputShape, batch.length);
            const targets = new Matrix(network.layers[network.layers.length - 1].outputShape, batch.length);

            // Prepare batch inputs and targets
            for (let j = 0; j < batch.length; j++) {
                inputs.setColumn(j, batch[j].input);
                targets.setColumn(j, batch[j].output);
            }

            const outputs = network.forward(inputs);
            const errors = outputs.add(targets, -1);

            network.backward(errors, learningRate);

            // Calculate error and accuracy for the batch
            let batchError = 0;
            let batchAcc = 0;
            for (let j = 0; j < batch.length; j++) {
                const error = errors.getColumn(j);
                const output = outputs.getColumn(j);
                const target = targets.getColumn(j);

                totalError += error.reduce((sum, val) => sum + val * val, 0);
                batchError += error.reduce((sum, val) => sum + val * val, 0);
                if (output.indexOf(Math.max(...output)) === target.indexOf(1)) {
                    accuracy++;
                    batchAcc++;
                }
            }
            batchError /= batch.length;
            batchAcc /= batch.length;

            // Update progress bar
            updateProgressBar(batchIdx, totalBatches, epoch, batchError, batchAcc);
            batchIdx++;
        }
        const endTime = Date.now();

        //console.log(`\nEpoch ${epoch + 1}, Error: ${totalError / trainingSet.length}, Accuracy: ${accuracy / trainingSet.length}`);
        updateProgressBar(batchIdx, totalBatches, epoch, totalError / trainingSet.length, accuracy / trainingSet.length);
        // Append time taken w/out console.log
        test(network, testSet);
        console.log(`, Time taken: ${(endTime - startTime) / 1000}s, Images/sec: ${Math.round(trainingSet.length / ((endTime - startTime) / 1000))}`);
    }
}

function test(network, dataset) {
    let correct = 0;
    for (let data of dataset) {
        const input = new Matrix(network.layers[0].inputShape, 1).fromArray(data.input);
        const target = new Matrix(network.layers[network.layers.length - 1].outputShape, 1).fromArray(data.output);

        const output = network.forward(input);
        if (output.data.indexOf(Math.max(...output.data)) === target.data.indexOf(1)) {
            correct++;
        }
    }
    process.stdout.write(`, Test accuracy: ${(correct / dataset.length).toFixed(3)}`);
}


function main() {
    const network = new MLP([784, 64, 10]);
    train(network, 10, LEARNING_RATE / BATCH_SIZE, BATCH_SIZE);
}
main();