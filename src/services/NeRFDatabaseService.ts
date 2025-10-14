import Dexie, { Table } from 'dexie';
import { v4 as uuidv4 } from 'uuid';
import { SerializableWeights } from '../workers/NeRFModelWorker';

export interface ImageRecord {
  id: string;
  timestamp: number;
  filename: string;
  mimeType: string;
  data: Blob;
}

export interface CameraPoseRecord {
  id: string;
  imageId: string;
  poseMatrix: Float32Array;
  timestamp: number;
}

export interface ModelCheckpointRecord {
  id: string;
  modelVersion: number;
  timestamp: number;
  weights: SerializableWeights;
  trainingEpoch: number;
  loss: number;
  metrics?: { psnr?: number; ssim?: number };
}

class NeRFDatabase extends Dexie {
  images!: Table<ImageRecord, string>;
  cameraPoses!: Table<CameraPoseRecord, string>;
  modelCheckpoints!: Table<ModelCheckpointRecord, string>;

  constructor() {
    super('NeRFDatabase');
    this.version(1).stores({
      images: '++id, timestamp, filename',
      cameraPoses: '++id, imageId, timestamp',
      modelCheckpoints: '++id, modelVersion, timestamp'
    });
  }
}

const db = new NeRFDatabase();

export class NeRFDatabaseService {
  async saveImage(file: Blob | File, filename?: string): Promise<string> {
    const id = uuidv4();
    const record: ImageRecord = {
      id,
      timestamp: Date.now(),
      filename: filename || (file instanceof File ? file.name : `image-${id}`),
      mimeType: file.type,
      data: file
    };
    await db.images.add(record);
    return id;
  }

  async getImages(): Promise<ImageRecord[]> {
    return db.images.toArray();
  }

  async getImageById(id: string): Promise<ImageRecord | undefined> {
    return db.images.get(id);
  }

  async saveCameraPose(imageId: string, poseMatrix: Float32Array): Promise<string> {
    const id = uuidv4();
    const record: CameraPoseRecord = {
      id,
      imageId,
      poseMatrix,
      timestamp: Date.now()
    };
    await db.cameraPoses.add(record);
    return id;
  }

  async getCameraPoses(): Promise<CameraPoseRecord[]> {
    return db.cameraPoses.toArray();
  }

  async getCameraPosesByImageId(imageId: string): Promise<CameraPoseRecord[]> {
    return db.cameraPoses.where('imageId').equals(imageId).toArray();
  }

  async saveModelCheckpoint(
    weights: SerializableWeights,
    modelVersion: number,
    trainingEpoch: number,
    loss: number,
    metrics?: { psnr?: number; ssim?: number }
  ): Promise<string> {
    const id = uuidv4();
    const record: ModelCheckpointRecord = {
      id,
      modelVersion,
      timestamp: Date.now(),
      weights,
      trainingEpoch,
      loss,
      metrics
    };
    await db.modelCheckpoints.add(record);
    return id;
  }

  async getLatestModelCheckpoint(): Promise<ModelCheckpointRecord | undefined> {
    const maxVersionRecord = await db.modelCheckpoints
      .orderBy('modelVersion')
      .reverse()
      .first();

    if (!maxVersionRecord) {
      return undefined;
    }

    const maxModelVersion = maxVersionRecord.modelVersion;

    const latestCheckpoints = await db.modelCheckpoints
      .where('modelVersion')
      .equals(maxModelVersion)
      .toArray();

    if (latestCheckpoints.length > 0) {
      latestCheckpoints.sort((a, b) => b.timestamp - a.timestamp);
      return latestCheckpoints[0];
    }

    return undefined;
  }

  async clearTable(tableName: 'images' | 'cameraPoses' | 'modelCheckpoints'): Promise<void> {
    await db.table(tableName).clear();
  }

  async clearAllData(): Promise<void> {
    await Promise.all([
      db.images.clear(),
      db.cameraPoses.clear(),
      db.modelCheckpoints.clear()
    ]);
  }
}

export const nerfDatabaseService = new NeRFDatabaseService();