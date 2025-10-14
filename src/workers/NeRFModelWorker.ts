import * as tf from '@tensorflow/tfjs';
import * as Comlink from 'comlink';
import { nerfDataUtils } from '../utils/NeRFDataUtils';

const L_POS = 10;
const L_DIR = 4;

const N_SAMPLES = 64;
const NEAR_PLANE = 0.0;
const FAR_PLANE = 1.0;
const PERTURB_SAMPLES = true;

function positionalEncoding(x: tf.Tensor, L: number): tf.Tensor {
  return tf.tidy(() => {
    const freqsTensor = tf.pow(2, tf.range(0, L, 1, 'float32')).mul(Math.PI);
    const xExpanded = x.expandDims(-1);
    const sinComponents = tf.sin(xExpanded.mul(freqsTensor));
    const cosComponents = tf.cos(xExpanded.mul(freqsTensor));
    const encoded = tf.concat([
      xExpanded,
      sinComponents,
      cosComponents
    ], -1);
    const outputShape = x.shape.slice(0, x.shape.length - 1).concat([x.shape[x.shape.length - 1] * (1 + 2 * L)]);
    return encoded.reshape(outputShape);
  });
}

export interface SerializableWeights {
  weightData: ArrayBuffer;
  weightSpecs: tf.io.WeightsManifestEntry[];
}

interface TrainingMetrics {
  epoch: number;
  loss: number;
  metrics: { psnr: number; ssim: number };
}

interface RenderedImageData {
  imageData: ArrayBuffer; // Uint8Array.buffer
  width: number;
  height: number;
}

export class NeRFModelWorker {
  private model: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer | null = null;
  private L_pos: number;
  private L_dir: number;
  private lastTrainingMetrics: TrainingMetrics = { epoch: 0, loss: 0, metrics: { psnr: 0, ssim: 0 } };

  constructor(L_pos: number = L_POS, L_dir: number = L_DIR) {
    this.L_pos = L_pos;
    this.L_dir = L_dir;
  }

  async initModel(): Promise<void> {
    if (this.model) {
      console.warn('NeRF model already initialized. Disposing old model.');
      this.model.dispose();
      if (this.optimizer) {
        this.optimizer.dispose();
      }
    }

    const inputPointsDim = 3 * (1 + 2 * this.L_pos);
    const inputDirectionsDim = 3 * (1 + 2 * this.L_dir);

    const inputPoints = tf.input({ shape: [inputPointsDim], name: 'input_points' });
    const inputDirections = tf.input({ shape: [inputDirectionsDim], name: 'input_directions' });

    let x: tf.SymbolicTensor = inputPoints;
    for (let i = 0; i < 8; i++) {
      x = tf.layers.dense({ units: 256, activation: 'relu', name: `point_dense_${i}` }).apply(x) as tf.SymbolicTensor;
      if (i === 3) {
        x = tf.layers.concatenate({ name: 'skip_connection' }).apply([x, inputPoints]) as tf.SymbolicTensor;
      }
    }

    const sigma = tf.layers.dense({ units: 1, activation: 'relu', name: 'sigma_output' }).apply(x) as tf.SymbolicTensor;
    const feature = tf.layers.dense({ units: 256, activation: 'linear', name: 'feature_output' }).apply(x) as tf.SymbolicTensor;
    const combined = tf.layers.concatenate({ name: 'feature_direction_concat' }).apply([feature, inputDirections]) as tf.SymbolicTensor;

    let rgb = tf.layers.dense({ units: 128, activation: 'relu', name: 'direction_dense_0' }).apply(combined) as tf.SymbolicTensor;
    rgb = tf.layers.dense({ units: 3, activation: 'sigmoid', name: 'rgb_output' }).apply(rgb) as tf.SymbolicTensor;

    this.model = tf.model({ inputs: [inputPoints, inputDirections], outputs: [rgb, sigma] });
    this.optimizer = tf.train.adam(1e-4);

    console.log('NeRF model initialized.');
  }

  predict(points: tf.Tensor, viewDirections: tf.Tensor): { rgb: tf.Tensor; sigma: tf.Tensor } {
    if (!this.model) {
      throw new Error('Model not initialized. Call initModel() first.');
    }

    return tf.tidy(() => {
      const encodedPoints = positionalEncoding(points, this.L_pos);
      const encodedDirections = positionalEncoding(viewDirections, this.L_dir);

      const [rgb, sigma] = this.model!.predict([encodedPoints, encodedDirections]) as [tf.Tensor, tf.Tensor];

      return { rgb, sigma };
    });
  }

  async train(
    rayOrigins: tf.Tensor,
    rayDirections: tf.Tensor,
    targetRgb: tf.Tensor,
    near: number = NEAR_PLANE,
    far: number = FAR_PLANE,
    N_samples: number = N_SAMPLES,
    perturb: boolean = PERTURB_SAMPLES
  ): Promise<number> {
    if (!this.model) {
      throw new Error('Model not initialized. Call initModel() first.');
    }
    if (!this.optimizer) {
      throw new Error('Optimizer not initialized. Call initModel() first.');
    }

    let lossValue: number = 0;

    tf.tidy(() => {
      tf.util.assert(rayOrigins.shape.length === 2 && rayOrigins.shape[1] === 3, () => `Expected rayOrigins to be [N, 3], but got ${rayOrigins.shape}`);
      tf.util.assert(rayDirections.shape.length === 2 && rayDirections.shape[1] === 3, () => `Expected rayDirections to be [N, 3], but got ${rayDirections.shape}`);
      tf.util.assert(targetRgb.shape.length === 2 && targetRgb.shape[1] === 3, () => `Expected targetRgb to be [N, 3], but got ${targetRgb.shape}`);
      tf.util.assert(rayOrigins.shape[0] === targetRgb.shape[0], () => 'Batch sizes of ray origins and target RGB must match.');

      const N_rays = rayOrigins.shape[0];

      this.optimizer!.minimize(() => {
        const { sampledPoints, depthValues } = nerfDataUtils.samplePointsAlongRays(
          rayOrigins, rayDirections, near, far, N_samples, perturb
        );

        const viewDirectionsForSamples = tf.tile(rayDirections.expandDims(1), [1, N_samples, 1])
          .reshape([-1, 3]);

        const { rgb: predictedRgbSamples, sigma: predictedSigmaSamples } = this.predict(
          sampledPoints, viewDirectionsForSamples
        );

        const predictedRgbReshaped = predictedRgbSamples.reshape([N_rays, N_samples, 3]);
        const predictedSigmaReshaped = predictedSigmaSamples.reshape([N_rays, N_samples, 1]);

        const { renderedRgb } = nerfDataUtils.volumeRender(
          predictedRgbReshaped, predictedSigmaReshaped, depthValues
        );

        const loss = tf.losses.meanSquaredError(targetRgb, renderedRgb).mean();

        lossValue = loss.arraySync() as number;
        return loss as tf.Scalar;
      });
    });
    this.lastTrainingMetrics = {
      epoch: this.lastTrainingMetrics.epoch + 1,
      loss: lossValue,
      metrics: { psnr: this.lastTrainingMetrics.metrics.psnr, ssim: this.lastTrainingMetrics.metrics.ssim }
    };
    console.log(`NeRFModelWorker.train: Performed training step. Loss: ${lossValue.toFixed(6)}. Epoch: ${this.lastTrainingMetrics.epoch}`);
    return lossValue;
  }

  async getWeights(): Promise<SerializableWeights> {
    if (!this.model) {
      throw new Error('Model not initialized.');
    }
    const weights = this.model!.getWeights();
    const weightNames: string[] = [];
    for (const w of this.model!.weights) {
      weightNames.push(w.name);
    }

    const namedTensors = weights.map((tensor, i) => ({
      name: weightNames[i],
      tensor: tensor
    }));

    const { data, specs } = await tf.io.encodeWeights(namedTensors);
    const serializableWeights: SerializableWeights = {
      weightData: data,
      weightSpecs: specs
    };
    return Comlink.transfer(serializableWeights, [serializableWeights.weightData]);
  }

  async setWeights(weights: SerializableWeights): Promise<void> {
    if (!this.model) {
      throw new Error('Model not initialized.');
    }
    const weightTensors = await tf.io.decodeWeights(weights.weightData, weights.weightSpecs as tf.io.WeightsManifestEntry[]);
    this.model!.setWeights(Object.values(weightTensors));
  }

  private calculateChannelSSIM(x: tf.Tensor, y: tf.Tensor): number {
    return tf.tidy(() => {
      const L = 1.0;
      const K1 = 0.01;
      const K2 = 0.03;
      const C1 = (K1 * L) ** 2;
      const C2 = (K2 * L) ** 2;

      const mu_x = x.mean();
      const mu_y = y.mean();

      const sigma_x_sq = x.sub(mu_x).square().mean();
      const sigma_y_sq = y.sub(mu_y).square().mean();

      const sigma_xy = x.sub(mu_x).mul(y.sub(mu_y)).mean();

      const numerator = mu_x.mul(mu_y).mul(2).add(C1)
        .mul(sigma_xy.mul(2).add(C2));
      const denominator = mu_x.square().add(mu_y.square()).add(C1)
        .mul(sigma_x_sq.add(sigma_y_sq).add(C2));

      const denomVal = denominator.arraySync() as number;
      const numerVal = numerator.arraySync() as number;

      if (Math.abs(denomVal) < Number.EPSILON) {
        return (Math.abs((mu_x.arraySync() as number) - (mu_y.arraySync() as number)) < Number.EPSILON) ? 1.0 : 0.0;
      }

      return numerVal / denomVal;
    });
  }

  async calculateMetrics(validationData: { predictedRgb: tf.Tensor; groundTruthRgb: tf.Tensor }): Promise<{ psnr: number; ssim: number }> {
    if (!this.model) {
      throw new Error('Model not initialized. Call initModel() first.');
    }

    return tf.tidy(() => {
      const { predictedRgb, groundTruthRgb } = validationData;

      tf.util.assert(predictedRgb.shape.length === 2 && predictedRgb.shape[1] === 3, () => `Expected predictedRgb to be [N, 3], but got ${predictedRgb.shape}`);
      tf.util.assert(groundTruthRgb.shape.length === 2 && groundTruthRgb.shape[1] === 3, () => `Expected groundTruthRgb to be [N, 3], but got ${groundTruthRgb.shape}`);
      tf.util.assert(predictedRgb.shape.every((dim, i) => dim === groundTruthRgb.shape[i]), () => 'Predicted and ground truth RGB tensors must have the same shape.');

      const mse = tf.losses.meanSquaredError(groundTruthRgb, predictedRgb).mean().arraySync() as number;

      let psnr = 0;
      if (mse > 0) {
        const max_i = 1.0;
        psnr = 10 * Math.log10(max_i * max_i / mse);
      } else {
        psnr = Infinity;
      }

      let ssimSum = 0;
      for (let i = 0; i < 3; i++) {
        const predictedChannel = predictedRgb.slice([0, i], [-1, 1]).squeeze();
        const groundTruthChannel = groundTruthRgb.slice([0, i], [-1, 1]).squeeze();
        ssimSum += this.calculateChannelSSIM(predictedChannel, groundTruthChannel);
      }
      const ssim = ssimSum / 3;

      this.lastTrainingMetrics.metrics = { psnr, ssim };

      console.log(`NeRFModelWorker.calculateMetrics: Calculated PSNR: ${psnr.toFixed(4)}, SSIM: ${ssim.toFixed(4)} (global per-channel SSIM).`);
      return { psnr, ssim };
    });
  }

  async getTrainingMetrics(): Promise<TrainingMetrics> {
    return { ...this.lastTrainingMetrics };
  }

  async generateVoxelGrid(
    gridMin: number[],
    gridMax: number[],
    resolution: number
  ): Promise<{ voxelData: ArrayBuffer, resolution: number, gridMin: number[], gridMax: number[] }> {
    if (!this.model) {
      throw new Error('Model not initialized.');
    }
    if (gridMin.length !== 3 || gridMax.length !== 3) {
      throw new Error('gridMin and gridMax must be 3-element arrays.');
    }
    if (resolution <= 0 || !Number.isInteger(resolution)) {
      throw new Error('Resolution must be a positive integer.');
    }

    console.log(`Generating voxel grid with resolution ${resolution}x${resolution}x${resolution} from [${gridMin}] to [${gridMax}]...`);

    tf.engine().startScope();
    try {
      const xCoords = tf.linspace(gridMin[0], gridMax[0], resolution);
      const yCoords = tf.linspace(gridMin[1], gridMax[1], resolution);
      const zCoords = tf.linspace(gridMin[2], gridMax[2], resolution);

      const [gridX, gridY, gridZ] = (tf.meshgrid as any)(xCoords, yCoords, zCoords);
      const gridPoints = tf.stack([gridX.flatten(), gridY.flatten(), gridZ.flatten()], 1);
      
      const numPoints = resolution * resolution * resolution;
      const dummyViewDirections = tf.tile(tf.tensor2d([0, 0, 1], [1, 3]), [numPoints, 1]);

      const { rgb, sigma } = this.predict(gridPoints, dummyViewDirections);

      const combinedData = tf.concat([rgb, sigma], -1);

      const voxelDataArray = await combinedData.data();
      const flatVoxelData = new Float32Array(voxelDataArray);

      console.log('Voxel grid generation complete.');
      return Comlink.transfer(
        { voxelData: flatVoxelData.buffer, resolution, gridMin, gridMax },
        [flatVoxelData.buffer]
      );
    } finally {
      tf.engine().endScope();
    }
  }

  async renderNeRFImage(
    H: number, W: number,
    focalX: number, focalY: number,
    centerX: number, centerY: number,
    c2w: Float32Array,
    N_samples: number = N_SAMPLES,
    near: number = NEAR_PLANE,
    far: number = FAR_PLANE,
    perturb: boolean = PERTURB_SAMPLES
  ): Promise<RenderedImageData> {
    if (!this.model) {
      throw new Error('Model not initialized. Call initModel() first.');
    }

    try {
      tf.engine().startScope();
      const rays = nerfDataUtils.getRays(
        H, W,
        focalX, focalY,
        centerX, centerY,
        c2w
      );
      const sampled = nerfDataUtils.samplePointsAlongRays(
        rays.rayOrigins, rays.rayDirections, near, far, N_samples, perturb
      );
      const viewDirectionsForSamples = tf.tile(rays.rayDirections.expandDims(1), [1, N_samples, 1])
        .reshape([-1, 3]);
      const predictions = this.predict(
        sampled.sampledPoints, viewDirectionsForSamples
      );
      const N_rays = rays.rayOrigins.shape[0];
      const predictedRgbReshaped = predictions.rgb.reshape([N_rays, N_samples, 3]);
      const predictedSigmaReshaped = predictions.sigma.reshape([N_rays, N_samples, 1]);
      const volumeRendered = nerfDataUtils.volumeRender(
        predictedRgbReshaped, predictedSigmaReshaped, sampled.depthValues
      );
      const renderedRgb = volumeRendered.renderedRgb;

      const imageData = await renderedRgb.mul(255).cast('int32').data();
      const pixelArray = new Uint8Array(W * H * 4);
      for (let i = 0; i < imageData.length; i++) {
        const [r, g, b] = (imageData as any)[i];
        pixelArray[i * 4 + 0] = r;
        pixelArray[i * 4 + 1] = g;
        pixelArray[i * 4 + 2] = b;
        pixelArray[i * 4 + 3] = 255; // Alpha
      }

      return Comlink.transfer(
        { imageData: pixelArray.buffer, width: W, height: H },
        [pixelArray.buffer]
      );
    } catch (error) {
      console.error('Error during centralized NeRF rendering:', error);
      throw error;
    } finally {
      tf.engine().endScope();
    }
  }
}

Comlink.expose(new NeRFModelWorker());