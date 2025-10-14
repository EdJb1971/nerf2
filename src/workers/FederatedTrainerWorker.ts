import * as Comlink from 'comlink';
import { Peer, DataConnection } from 'peerjs';
import * as msgpack from '@msgpack/msgpack';
import * as tf from '@tensorflow/tfjs';
import { nerfDatabaseService } from '../services/NeRFDatabaseService';
import { NeRFModelWorker, SerializableWeights } from './NeRFModelWorker';

// Helper to convert serializable weights to tf.Tensor[]
async function serializableWeightsToTensors(serializable: SerializableWeights): Promise<tf.Tensor[]> {
  const a = await tf.io.decodeWeights(serializable.weightData, serializable.weightSpecs as tf.io.WeightsManifestEntry[]);
  return Object.values(a);
}

// Helper to convert tf.Tensor[] to serializable weights
async function tensorsToSerializableWeights(tensors: tf.Tensor[]): Promise<SerializableWeights> {
    const namedTensors = tensors.map((tensor, i) => ({
        name: `weight${i}`,
        tensor: tensor,
    }));
    const modelArtifacts = await tf.io.encodeWeights(namedTensors);
    return {
      weightData: modelArtifacts.data,
      weightSpecs: modelArtifacts.specs,
    };
}

interface FederatedSchedule {
  iterationsPerRound: number;
  aggregationIntervalMs: number;
  minPeersForAggregation: number;
}

interface TrainingMetrics {
  epoch: number;
  loss: number;
  metrics: { psnr: number; ssim: number };
}

export class FederatedTrainerWorker {
  private peer: Peer | null = null;
  private connections: Map<string, DataConnection> = new Map();
  private nerfModelWorker: Comlink.Remote<NeRFModelWorker> | null = null;
  // Buffer now stores SerializableWeights (deltas)
  private receivedWeightsBuffer: Map<string, SerializableWeights> = new Map();
  private currentPeerId: string | null = null;
  private nerfDatabaseService: typeof nerfDatabaseService;
  private sharedSecret: string | null = null;

  private noiseMultiplier: number = 0.001;
  private clipNorm: number = 0.5;

  private aggregationTimer: NodeJS.Timeout | null = null;
  private localTrainingIterationCount: number = 0;

  // New: Store the last aggregated global model weights
  private lastAggregatedWeights: SerializableWeights | null = null;

  // CQ-FTW-001: Configurable communication schedule parameters
  private iterationsPerRound: number = 5;
  private aggregationIntervalMs: number = 10000;
  private minPeersForAggregation: number = 1;

  constructor(dbService: typeof nerfDatabaseService) {
    this.nerfDatabaseService = dbService;
  }

  async initNeRFModelWorker(worker: Comlink.Remote<NeRFModelWorker>): Promise<void> {
    this.nerfModelWorker = worker;
    console.log('FederatedTrainerWorker: NeRFModelWorker proxy initialized.');

    try {
      const latestCheckpoint = await this.nerfDatabaseService.getLatestModelCheckpoint();
      if (latestCheckpoint) {
        this.lastAggregatedWeights = latestCheckpoint.weights;
        console.log('FederatedTrainerWorker: Initialized lastAggregatedWeights from database checkpoint.');
      } else {
        const initialModelWeights = await this.nerfModelWorker.getWeights();
        this.lastAggregatedWeights = initialModelWeights;
        console.log('FederatedTrainerWorker: Initialized lastAggregatedWeights from fresh NeRF model.');
      }
    } catch (error) {
      console.error('Error loading or initializing lastAggregatedWeights for FederatedTrainerWorker:', error);
      if (!this.lastAggregatedWeights && this.nerfModelWorker) {
        this.lastAggregatedWeights = await this.nerfModelWorker.getWeights();
      }
    }
  }

  async startSession(peerId: string, sharedSecret: string, peerServerConfig?: { host: string; port: number; path: string }): Promise<string> {
    if (this.peer && this.peer.open) {
      console.warn('PeerJS session already active. Disconnecting old session.');
      this.peer.destroy();
      this.connections.clear();
      this.receivedWeightsBuffer.clear();
    }

    this.currentPeerId = peerId;
    this.sharedSecret = sharedSecret;
    this.peer = new Peer(peerId, peerServerConfig);

    return new Promise((resolve, reject) => {
      this.peer!.on('open', (id) => {
        console.log('PeerJS connection opened with ID:', id);
        resolve(id);
      });

      this.peer!.on('connection', (conn) => {
        console.log('Incoming connection from:', conn.peer);
        if (conn.metadata && (conn.metadata as any).sharedSecret === this.sharedSecret) {
          console.log(`Authenticated incoming connection from: ${conn.peer}`);
          this.setupConnectionListeners(conn);
        } else {
          console.warn(`Unauthorized incoming connection from ${conn.peer}. Closing connection.`);
          conn.close();
        }
      });

      this.peer!.on('error', (err) => {
        console.error('PeerJS error:', err);
        reject(err);
      });

      this.peer!.on('disconnected', () => {
        console.log('PeerJS disconnected.');
        this.connections.clear();
        this.receivedWeightsBuffer.clear();
        this.stopAggregationTimer();
      });

      this.peer!.on('close', () => {
        console.log('PeerJS connection closed.');
        this.connections.clear();
        this.receivedWeightsBuffer.clear();
        this.stopAggregationTimer();
      });
    });
  }

  private setupConnectionListeners(conn: DataConnection): void {
    this.connections.set(conn.peer, conn);

    conn.on('data', (data) => {
      if (typeof data === 'object' && data !== null && 'metadata' in data && 'weightData' in data && data.weightData instanceof ArrayBuffer) {
        console.log(`Received weights from peer: ${conn.peer}`);
        this.receiveWeightsFromPeer(conn.peer, data as { metadata: ArrayBuffer; weightData: ArrayBuffer });
      } else {
        console.warn(`Received unexpected data format from ${conn.peer}. Ignoring.`);
      }
    });

    conn.on('open', () => {
      console.log(`Data connection opened with peer: ${conn.peer}`);
    });

    conn.on('close', () => {
      console.log(`Data connection closed with peer: ${conn.peer}`);
      this.connections.delete(conn.peer);
      this.receivedWeightsBuffer.delete(conn.peer);
    });

    conn.on('error', (err) => {
      console.error(`Data connection error with peer ${conn.peer}:`, err);
      this.connections.delete(conn.peer);
      this.receivedWeightsBuffer.delete(conn.peer);
    });
  }

  async connectToPeer(targetPeerId: string): Promise<void> {
    if (!this.peer || !this.peer.open) {
      throw new Error('PeerJS session not started. Call startSession() first.');
    }
    if (this.connections.has(targetPeerId)) {
      console.log(`Already connected to peer: ${targetPeerId}`);
      return;
    }
    if (this.currentPeerId === targetPeerId) {
      console.warn('Cannot connect to self.');
      return;
    }
    if (!this.sharedSecret) {
      throw new Error('Shared secret not set. Call startSession() with a sharedSecret.');
    }

    console.log(`Attempting to connect to peer: ${targetPeerId}`);
    const conn = this.peer.connect(targetPeerId, { metadata: { sharedSecret: this.sharedSecret } });
    this.setupConnectionListeners(conn);
  }

  async sendWeightsToPeers(): Promise<void> {
    if (!this.nerfModelWorker) {
      throw new Error('NeRFModelWorker proxy not initialized.');
    }
    if (this.connections.size === 0) {
      console.warn('No peers connected to send weights to.');
      return;
    }
    if (!this.lastAggregatedWeights) {
      console.error('lastAggregatedWeights is not initialized. Cannot calculate delta. Aborting send.');
      return;
    }

    console.log('Calculating delta weights, applying differential privacy, and sending to peers...');

    tf.engine().startScope();
    try {
      const currentLocalWeights = await this.nerfModelWorker!.getWeights();
      const currentLocalTensors = await serializableWeightsToTensors(currentLocalWeights);
      const lastAggregatedTensors = await serializableWeightsToTensors(this.lastAggregatedWeights!);

      if (currentLocalTensors.length !== lastAggregatedTensors.length ||
          !currentLocalTensors.every((t, i) => tf.util.arraysEqual(t.shape, lastAggregatedTensors[i].shape))) {
        console.error('Shape mismatch between current local weights and last aggregated weights. Cannot calculate delta. Aborting send.');
        return;
      }

      const deltaTensors = currentLocalTensors.map((localT, i) => localT.sub(lastAggregatedTensors[i]));
      const privatizedDeltaTensors = this.addDifferentialPrivacy(deltaTensors, this.noiseMultiplier, this.clipNorm);
      const privatizedSerializableDelta = await tensorsToSerializableWeights(privatizedDeltaTensors);

      const payload = {
        weightSpecs: privatizedSerializableDelta.weightSpecs
      };
      const serializedPayload = msgpack.encode(payload);

      for (const conn of this.connections.values()) {
        if (conn.open) {
          conn.send(Comlink.transfer({
            metadata: serializedPayload,
            weightData: privatizedSerializableDelta.weightData
          }, [privatizedSerializableDelta.weightData]));
          console.log(`Sent delta weights to peer: ${conn.peer}`);
        }
      }
    } finally {
      tf.engine().endScope();
    }
  }

  private addDifferentialPrivacy(weights: tf.Tensor[], noiseMultiplier: number, clipNorm: number): tf.Tensor[] {
    return tf.tidy(() => {
      return weights.map(w => {
        const clippedWeight = tf.clipByValue(w, -clipNorm, clipNorm);
        const noise = tf.randomNormal(clippedWeight.shape, 0, noiseMultiplier);
        return clippedWeight.add(noise);
      });
    });
  }

  async receiveWeightsFromPeer(peerId: string, data: { metadata: ArrayBuffer; weightData: ArrayBuffer }): Promise<void> {
    try {
      const decodedPayload = msgpack.decode(data.metadata) as { weightSpecs: tf.io.WeightsManifestEntry[] };
      const receivedSerializableWeights: SerializableWeights = {
        weightData: data.weightData,
        weightSpecs: decodedPayload.weightSpecs
      };
      const decodedTensors = await serializableWeightsToTensors(receivedSerializableWeights);

      if (!this.nerfModelWorker) {
        throw new Error('NeRFModelWorker proxy not initialized for Byzantine detection.');
      }
      const localSerializableWeights = await this.nerfModelWorker!.getWeights();
      const localTensorsForShapeCheck = await serializableWeightsToTensors(localSerializableWeights);
      const expectedShapes = localTensorsForShapeCheck.map(t => t.shape);

      if (this.byzantineDetection(decodedTensors, expectedShapes)) {
        this.receivedWeightsBuffer.set(peerId, receivedSerializableWeights);
        console.log(`Delta weights from ${peerId} buffered for aggregation.`);
      } else {
        console.warn(`Byzantine detection failed for delta weights from ${peerId}. Discarding.`);
      }
      tf.dispose(decodedTensors);
      tf.dispose(localTensorsForShapeCheck);
    } catch (error) {
      console.error(`Error decoding or processing delta weights from ${peerId}:`, error);
    }
  }

  private byzantineDetection(weights: tf.Tensor[], expectedShapes: number[][]): boolean {
    if (!Array.isArray(weights) || weights.length === 0) {
      console.warn('Byzantine detection: Received empty or invalid weights array.');
      return false;
    }
    if (weights.length !== expectedShapes.length) {
      console.warn(`Byzantine detection: Received weights array length (${weights.length}) does not match expected length (${expectedShapes.length}).`);
      return false;
    }

    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      const expectedShape = expectedShapes[i];

      if (!(w instanceof tf.Tensor)) {
        console.warn('Byzantine detection: Received non-tensor element in weights array.');
        return false;
      }
      if (!tf.util.arraysEqual(w.shape, expectedShape)) {
        console.warn(`Byzantine detection: Weight tensor at index ${i} has inconsistent shape. Expected ${JSON.stringify(expectedShape)}, got ${JSON.stringify(w.shape)}.`);
        return false;
      }
      if (tf.any(w.isNaN()).arraySync() || tf.any(w.isInf()).arraySync()) {
        console.warn('Byzantine detection: Received weights contain NaN or Infinity values.');
        return false;
      }
    }
    return true;
  }

  async incrementLocalTrainingIteration(): Promise<number> {
    this.localTrainingIterationCount++;
    console.log(`Local training iteration: ${this.localTrainingIterationCount}`);
    return this.localTrainingIterationCount;
  }

  async getLocalTrainingIterationCount(): Promise<number> {
    return this.localTrainingIterationCount;
  }

  async setCommunicationSchedule(schedule: FederatedSchedule): Promise<void> {
    this.iterationsPerRound = schedule.iterationsPerRound;
    this.aggregationIntervalMs = schedule.aggregationIntervalMs;
    this.minPeersForAggregation = schedule.minPeersForAggregation;
    console.log('FederatedTrainerWorker: Communication schedule updated.', schedule);
  }

  async startAggregationTimer(): Promise<void> {
    if (this.aggregationTimer) {
      console.warn('Aggregation timer already running. Stopping existing timer.');
      this.stopAggregationTimer();
    }

    console.log(`Starting aggregation timer with interval: ${this.aggregationIntervalMs}ms`);

    this.aggregationTimer = setInterval(async () => {
      console.log('Attempting federated aggregation...');
      try {
        await this.aggregateWeights();
      } catch (error) {
        console.error('Error during scheduled aggregation:', error);
      }
    }, this.aggregationIntervalMs);
  }

  async stopAggregationTimer(): Promise<void> {
    if (this.aggregationTimer) {
      clearInterval(this.aggregationTimer);
      this.aggregationTimer = null;
      console.log('Aggregation timer stopped.');
    }
  }

  async aggregateWeights(): Promise<void> {
    if (!this.nerfModelWorker) {
      throw new Error('NeRFModelWorker proxy not initialized.');
    }
    if (!this.lastAggregatedWeights) {
      throw new Error('lastAggregatedWeights is not initialized. Cannot perform delta aggregation.');
    }

    if (this.receivedWeightsBuffer.size < this.minPeersForAggregation) {
      console.warn(`Not enough peer delta weights (${this.receivedWeightsBuffer.size}) for aggregation. Minimum required: ${this.minPeersForAggregation}. Skipping aggregation.`);
      return;
    }

    console.log('Starting federated delta aggregation...');

    tf.engine().startScope();
    try {
      const baseGlobalModelTensors = await serializableWeightsToTensors(this.lastAggregatedWeights!);

      const currentLocalSerializableWeights = await this.nerfModelWorker!.getWeights();
      const currentLocalTensors = await serializableWeightsToTensors(currentLocalSerializableWeights);

      const localDeltaTensors = currentLocalTensors.map((localT, i) => localT.sub(baseGlobalModelTensors[i]));

      const allDeltasTensors: tf.Tensor[][] = [localDeltaTensors];
      for (const peerSerializableDelta of this.receivedWeightsBuffer.values()) {
        allDeltasTensors.push(await serializableWeightsToTensors(peerSerializableDelta));
      }

      if (allDeltasTensors.length === 0) {
        console.warn('No deltas available for aggregation.');
        return;
      }

      const numDeltaSets = allDeltasTensors.length;
      const numTensors = baseGlobalModelTensors.length;
      for (let i = 0; i < numDeltaSets; i++) {
        if (allDeltasTensors[i].length !== numTensors) {
          console.error('Delta weight sets have inconsistent number of tensors. Skipping aggregation.');
          return;
        }
        for (let j = 0; j < numTensors; j++) {
          if (!tf.util.arraysEqual(allDeltasTensors[i][j].shape, baseGlobalModelTensors[j].shape)) {
            console.error(`Delta tensor ${j} has inconsistent shape across peers. Expected ${JSON.stringify(baseGlobalModelTensors[j].shape)}, got ${JSON.stringify(allDeltasTensors[i][j].shape)}. Skipping aggregation.`);
            return;
          }
        }
      }

      const averagedDeltaTensors: tf.Tensor[] = [];
      for (let i = 0; i < numTensors; i++) {
        let sumDeltaTensor = tf.zerosLike(baseGlobalModelTensors[i]);
        for (let j = 0; j < numDeltaSets; j++) {
          sumDeltaTensor = sumDeltaTensor.add(allDeltasTensors[j][i]);
        }
        averagedDeltaTensors.push(sumDeltaTensor.div(numDeltaSets));
      }

      const newAggregatedGlobalTensors: tf.Tensor[] = baseGlobalModelTensors.map((baseT, i) =>
        baseT.add(averagedDeltaTensors[i])
      );

      const newAggregatedSerializableWeights = await tensorsToSerializableWeights(newAggregatedGlobalTensors);

      await this.nerfModelWorker!.setWeights(Comlink.transfer(newAggregatedSerializableWeights, [newAggregatedSerializableWeights.weightData]));
      console.log('Federated delta aggregation complete. Model updated.');

      this.lastAggregatedWeights = newAggregatedSerializableWeights;

      const trainingMetrics: TrainingMetrics = await this.nerfModelWorker!.getTrainingMetrics();

      const latestCheckpoint = await this.nerfDatabaseService.getLatestModelCheckpoint();
      const newModelVersion = latestCheckpoint ? latestCheckpoint.modelVersion + 1 : 1;

      await this.nerfDatabaseService.saveModelCheckpoint(
        newAggregatedSerializableWeights,
        newModelVersion,
        trainingMetrics.epoch,
        trainingMetrics.loss,
        trainingMetrics.metrics
      );
      console.log(`Aggregated model checkpoint saved with version: ${newModelVersion}.`);
    } finally {
        tf.engine().endScope();
    }

    this.receivedWeightsBuffer.clear();
  }

  async getCommunicationSchedule(): Promise<FederatedSchedule> {
    return {
      iterationsPerRound: this.iterationsPerRound,
      aggregationIntervalMs: this.aggregationIntervalMs,
      minPeersForAggregation: this.minPeersForAggregation
    };
  }
}

Comlink.expose(new FederatedTrainerWorker(nerfDatabaseService));
