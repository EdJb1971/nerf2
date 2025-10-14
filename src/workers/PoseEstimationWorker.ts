import * as Comlink from 'comlink';
import { mat4, vec3, quat } from 'gl-matrix';
import { nerfDatabaseService } from '../services/NeRFDatabaseService';
import { coordinateTransformer } from '../utils/CoordinateTransformer';

// Declare cv global for OpenCV.js
// This is loaded via importScripts in the worker
// and is not available at build time.
// @ts-ignore
declare const cv: any;

// Helper to load OpenCV.js
async function loadOpenCV(opencvPath: string): Promise<void> {
  return new Promise(async (resolve, reject) => {
    // @ts-ignore
    if (typeof cv !== 'undefined' && cv.Mat) {
      console.log('OpenCV.js already loaded.');
      resolve();
      return;
    }

    try {
      // In a module worker, importScripts is not available.
      // We must fetch the script and evaluate it in the worker's global scope.
      const response = await fetch(opencvPath);
      if (!response.ok) {
        reject(new Error(`Failed to fetch OpenCV.js: ${response.statusText}`));
        return;
      }
      const scriptText = await response.text();
      self.eval(scriptText);

      // The opencv.js script defines a global 'cv' object with an
      // 'onRuntimeInitialized' callback. We need to wait for that to be called.
      const checkCv = setInterval(() => {
        // @ts-ignore
        if (typeof cv !== 'undefined' && cv.onRuntimeInitialized) {
          // @ts-ignore
          cv.onRuntimeInitialized = () => {
            console.log('OpenCV.js loaded and initialized.');
            clearInterval(checkCv);
            resolve();
          };
        } else if (typeof cv !== 'undefined') {
            // If onRuntimeInitialized is not present, assume it's ready
            console.log('OpenCV.js loaded (assumed ready).');
            clearInterval(checkCv);
            resolve();
        }
      }, 100);

    } catch (error) {
      reject(error);
    }
  });
}

// Interface for image data when transferred to worker (Blob converted to ArrayBuffer)
interface TransferableImageRecord {
  id: string;
  filename: string;
  mimeType: string;
  data: ArrayBuffer;
}

export interface PhotoValidationResult {
  imageId: string;
  filename: string;
  brightness: number; // Mean pixel intensity (0-255)
  sharpness: number; // Variance of Laplacian
  redundancyScore?: number; // Max match ratio with another image in batch
  redundantWith?: string; // ID of the image it's redundant with
  isValid: boolean;
  messages: string[];
}

export class PoseEstimationWorker {
  private isCvLoaded: boolean = false;

  // Validation thresholds (CQ-004: Externalized hardcoded values)
  private BRIGHTNESS_MIN_THRESHOLD = 50;
  private BRIGHTNESS_MAX_THRESHOLD = 200;
  private SHARPNESS_THRESHOLD = 100;
  private REDUNDANCY_MATCH_RATIO_THRESHOLD = 0.8;

  constructor() {
    this.init();
  }

  private async init() {
    try {
      await loadOpenCV('/opencv.js');
      this.isCvLoaded = true;
      console.log('PoseEstimationWorker: OpenCV.js ready.');
    } catch (error) {
      console.error('PoseEstimationWorker: Failed to load OpenCV.js', error);
      this.isCvLoaded = false;
    }
  }

  private ensureCvLoaded(): void {
    if (!this.isCvLoaded) {
      throw new Error('OpenCV.js is not loaded or initialized. Call init() first.');
    }
  }

  async estimateRelativePoses(
    image1Id: string,
    image2Id: string,
    cameraMatrix: Float32Array,
    distCoeffs?: Float32Array
  ): Promise<{ image1Id: string; image2Id: string; poseMatrix: Float32Array }> {
    this.ensureCvLoaded();
    console.log(`Estimating relative poses for ${image1Id} and ${image2Id}`);

    if (!cameraMatrix || cameraMatrix.length !== 9) {
      throw new Error('Invalid cameraMatrix: Must be a Float32Array of 9 elements (3x3 matrix).');
    }
    if (distCoeffs && (distCoeffs.length !== 4 && distCoeffs.length !== 5)) {
      throw new Error('Invalid distCoeffs: Must be a Float32Array of 4 or 5 elements.');
    }

    const img1Record = await nerfDatabaseService.getImageById(image1Id);
    const img2Record = await nerfDatabaseService.getImageById(image2Id);

    if (!img1Record || !img2Record) {
      throw new Error('One or both images not found in the database.');
    }

    const camMat = cv.matFromArray(3, 3, cv.CV_32F, Array.from(cameraMatrix));
    const distCoeffsMat = distCoeffs ? cv.matFromArray(1, distCoeffs.length, cv.CV_32F, Array.from(distCoeffs)) : new cv.Mat();

    let img1 = new cv.Mat();
    let img2 = new cv.Mat();
    let gray1 = new cv.Mat();
    let gray2 = new cv.Mat();
    let keypoints1 = new cv.KeyPointVector();
    let keypoints2 = new cv.KeyPointVector();
    let descriptors1 = new cv.Mat();
    let descriptors2 = new cv.Mat();
    let matches = new cv.DMatchVector();
    let goodMatches = new cv.DMatchVector();
    let points1 = new cv.Mat();
    let points2 = new cv.Mat();
    let essentialMat = new cv.Mat();
    let R = new cv.Mat();
    let t = new cv.Mat();
    let mask = new cv.Mat();
    let orb: any;
    let bf: any;

    try {
      const img1Data = await img1Record.data.arrayBuffer();
      img1 = cv.imdecode(new cv.Mat(1, img1Data.byteLength, cv.CV_8U, new Uint8Array(img1Data)), cv.IMREAD_COLOR);
      const img2Data = await img2Record.data.arrayBuffer();
      img2 = cv.imdecode(new cv.Mat(1, img2Data.byteLength, cv.CV_8U, new Uint8Array(img2Data)), cv.IMREAD_COLOR);

      if (img1.empty() || img2.empty()) {
        throw new Error('Failed to decode images.');
      }

      cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY, 0);
      cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY, 0);

      orb = new cv.ORB();
      orb.detectAndCompute(gray1, new cv.Mat(), keypoints1, descriptors1);
      orb.detectAndCompute(gray2, new cv.Mat(), keypoints2, descriptors2);

      if (descriptors1.empty() || descriptors2.empty()) {
        throw new Error('No descriptors found for one or both images.');
      }

      bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
      bf.match(descriptors1, descriptors2, matches);

      const numMatches = matches.size();
      if (numMatches < 8) {
        throw new Error(`Not enough matches (${numMatches}) found for essential matrix estimation.`);
      }

      const matchesArray: any[] = [];
      for (let i = 0; i < numMatches; i++) {
        matchesArray.push(matches.get(i));
      }
      matchesArray.sort((a, b) => a.distance - b.distance);

      for (let i = 0; i < Math.min(200, numMatches * 0.5); i++) {
        goodMatches.push_back(matchesArray[i]);
      }

      if (goodMatches.size() < 8) {
        throw new Error(`Not enough good matches (${goodMatches.size()}) after filtering for essential matrix estimation.`);
      }

      points1 = new cv.Mat(goodMatches.size(), 2, cv.CV_32F);
      points2 = new cv.Mat(goodMatches.size(), 2, cv.CV_32F);

      for (let i = 0; i < goodMatches.size(); i++) {
        const match = goodMatches.get(i);
        const kp1 = keypoints1.get(match.queryIdx).pt;
        const kp2 = keypoints2.get(match.trainIdx).pt;
        points1.data32F[i * 2] = kp1.x;
        points1.data32F[i * 2 + 1] = kp1.y;
        points2.data32F[i * 2] = kp2.x;
        points2.data32F[i * 2 + 1] = kp2.y;
      }

      essentialMat = cv.findEssentialMat(points1, points2, camMat, cv.RANSAC, 0.999, 1.0, mask);

      if (essentialMat.empty()) {
        throw new Error('Failed to estimate Essential Matrix.');
      }

      cv.recoverPose(essentialMat, points1, points2, camMat, R, t, mask);

      if (R.empty() || t.empty()) {
        throw new Error('Failed to recover pose from Essential Matrix.');
      }

      const R_array: number[] = Array.from(R.data32F);
      const t_array: number[] = Array.from(t.data32F);

      const poseMatrix = coordinateTransformer.opencvRTToCameraMatrix(R_array, t_array);

      await nerfDatabaseService.saveCameraPose(image2Id, new Float32Array(poseMatrix));
      console.log(`Saved relative pose for image ${image2Id} (relative to ${image1Id}).`);

      return { image1Id, image2Id, poseMatrix: new Float32Array(poseMatrix) };

    } catch (error) {
      console.error('Error during pose estimation:', error);
      throw error;
    } finally {
      img1.delete();
      img2.delete();
      gray1.delete();
      gray2.delete();
      keypoints1.delete();
      keypoints2.delete();
      descriptors1.delete();
      descriptors2.delete();
      matches.delete();
      goodMatches.delete();
      points1.delete();
      points2.delete();
      essentialMat.delete();
      R.delete();
      t.delete();
      mask.delete();
      camMat.delete();
      distCoeffsMat.delete();
      if(orb) orb.delete();
      if(bf) bf.delete();
    }
  }

  async integrateDeviceOrientation(
    imageId: string,
    alpha: number,
    beta: number,
    gamma: number
  ): Promise<{ imageId: string; poseMatrix: Float32Array }> {
    console.log(`Integrating device orientation for image ${imageId}: alpha=${alpha}, beta=${beta}, gamma=${gamma}`);

    const poseMatrix = coordinateTransformer.deviceOrientationToCameraMatrix(alpha, beta, gamma);

    await nerfDatabaseService.saveCameraPose(imageId, new Float32Array(poseMatrix));
    console.log(`Saved device orientation pose for image ${imageId}.`);

    return { imageId, poseMatrix: new Float32Array(poseMatrix) };
  }

  async triangulatePoints(
    points1: Float32Array,
    points2: Float32Array,
    P1: Float32Array,
    P2: Float32Array
  ): Promise<Float32Array> {
    this.ensureCvLoaded();
    console.log('Triangulating 3D points from 2D correspondences.');

    if (!points1 || points1.length % 2 !== 0 || points1.length === 0) {
      throw new Error('Invalid points1: Must be a non-empty Float32Array with an even number of elements (N*2).');
    }
    if (!points2 || points2.length % 2 !== 0 || points2.length === 0) {
      throw new Error('Invalid points2: Must be a non-empty Float32Array with an even number of elements (N*2).');
    }
    if (points1.length !== points2.length) {
      throw new Error('points1 and points2 must have the same number of elements.');
    }
    if (!P1 || P1.length !== 12) {
      throw new Error('Invalid P1: Must be a Float32Array of 12 elements (3x4 projection matrix).');
    }
    if (!P2 || P2.length !== 12) {
      throw new Error('Invalid P2: Must be a Float32Array of 12 elements (3x4 projection matrix).');
    }

    const numPoints = points1.length / 2;

    let points1_mat = new cv.Mat(numPoints, 2, cv.CV_32F, points1);
    let points2_mat = new cv.Mat(numPoints, 2, cv.CV_32F, points2);
    let P1_mat = new cv.Mat(3, 4, cv.CV_32F, P1);
    let P2_mat = new cv.Mat(3, 4, cv.CV_32F, P2);
    let points4D = new cv.Mat();

    try {
      cv.triangulatePoints(P1_mat, P2_mat, points1_mat, points2_mat, points4D);

      const triangulatedPoints = new Float32Array(numPoints * 3);
      for (let i = 0; i < numPoints; i++) {
        const x = points4D.data32F[i * 4];
        const y = points4D.data32F[i * 4 + 1];
        const z = points4D.data32F[i * 4 + 2];
        const w = points4D.data32F[i * 4 + 3];

        if (Math.abs(w) < 1e-6) {
          triangulatedPoints[i * 3] = NaN;
          triangulatedPoints[i * 3 + 1] = NaN;
          triangulatedPoints[i * 3 + 2] = NaN;
          console.warn(`Triangulated point ${i} has a near-zero w-component, indicating a point at infinity or invalid triangulation.`);
        } else {
          triangulatedPoints[i * 3] = x / w;
          triangulatedPoints[i * 3 + 1] = y / w;
          triangulatedPoints[i * 3 + 2] = z / w;
        }
      }
      console.log(`Successfully triangulated ${numPoints} 3D points.`);
      return triangulatedPoints;

    } catch (error) {
      console.error('Error during 3D point triangulation:', error);
      throw error;
    } finally {
      points1_mat.delete();
      points2_mat.delete();
      P1_mat.delete();
      P2_mat.delete();
      points4D.delete();
    }
  }

  async refineGlobalPoses(poses: { imageId: string; poseMatrix: Float32Array }[]): Promise<{ imageId: string; poseMatrix: Float32Array }[]> {
    console.log(`Refining ${poses.length} global poses using iterative smoothing.`);
    console.warn('Note: This is NOT full Bundle Adjustment. For production use, consider:');
    console.warn('- COLMAP integration for robust BA');
    console.warn('- Ceres Solver via WebAssembly');
    console.warn('- External pre-processing of poses');

    if (poses.length <= 1) {
      return poses;
    }

    const refinedPoses = poses.map(p => ({
      imageId: p.imageId,
      poseMatrix: mat4.clone(p.poseMatrix)
    }));

    const numIterations = 5;
    const smoothingFactor = 0.2;

    for (let iter = 0; iter < numIterations; iter++) {
      const currentIterationPoses = refinedPoses.map(p => ({
        imageId: p.imageId,
        poseMatrix: mat4.clone(p.poseMatrix)
      }));

      for (let i = 0; i < refinedPoses.length; i++) {
        if (i === 0) continue;

        const currentPoseMat = currentIterationPoses[i].poseMatrix;
        let currentTranslation = vec3.create();
        let currentRotation = quat.create();
        mat4.getTranslation(currentTranslation, currentPoseMat);
        mat4.getRotation(currentRotation, currentPoseMat);

        let neighborTranslations: vec3[] = [];
        let neighborRotations: quat[] = [];

        const prevPoseMat = currentIterationPoses[i - 1].poseMatrix;
        let prevTranslation = vec3.create();
        let prevRotation = quat.create();
        mat4.getTranslation(prevTranslation, prevPoseMat);
        mat4.getRotation(prevRotation, prevPoseMat);
        neighborTranslations.push(prevTranslation);
        neighborRotations.push(prevRotation);

        if (i + 1 < refinedPoses.length) {
          const nextPoseMat = currentIterationPoses[i + 1].poseMatrix;
          let nextTranslation = vec3.create();
          let nextRotation = quat.create();
          mat4.getTranslation(nextTranslation, nextPoseMat);
          mat4.getRotation(nextRotation, nextPoseMat);
          neighborTranslations.push(nextTranslation);
          neighborRotations.push(nextRotation);
        }

        if (neighborTranslations.length > 0) {
          let avgNeighborTranslation = vec3.create();
          for (const t of neighborTranslations) {
            vec3.add(avgNeighborTranslation, avgNeighborTranslation, t);
          }
          vec3.scale(avgNeighborTranslation, avgNeighborTranslation, 1 / neighborTranslations.length);

          let avgNeighborRotation = quat.clone(neighborRotations[0]);
          for (let j = 1; j < neighborRotations.length; j++) {
            quat.slerp(avgNeighborRotation, avgNeighborRotation, neighborRotations[j], 1 / (j + 1));
          }
          quat.normalize(avgNeighborRotation, avgNeighborRotation);

          let newTranslation = vec3.create();
          vec3.lerp(newTranslation, currentTranslation, avgNeighborTranslation, smoothingFactor);

          let newRotation = quat.create();
          quat.slerp(newRotation, currentRotation, avgNeighborRotation, smoothingFactor);

          let newPoseMat = mat4.create();
          mat4.fromRotationTranslation(newPoseMat, newRotation, newTranslation);

          mat4.copy(refinedPoses[i].poseMatrix, newPoseMat);
        }
      }
    }

    console.log('PoseEstimationWorker.refineGlobalPoses: Completed basic iterative smoothing for global poses. This is NOT a full Bundle Adjustment and serves as a placeholder for more sophisticated optimization. It assumes a sequential order of poses for neighbor identification.');
    return refinedPoses.map(p => ({ imageId: p.imageId, poseMatrix: new Float32Array(p.poseMatrix) }));
  }

  async validateBatchPhotos(transferableImageRecords: TransferableImageRecord[]): Promise<PhotoValidationResult[]> {
    this.ensureCvLoaded();
    console.log(`Validating batch of ${transferableImageRecords.length} photos...`);

    const results: PhotoValidationResult[] = [];
    const imageMats: Map<string, any> = new Map();
    const grayMats: Map<string, any> = new Map();
    const descriptorsMap: Map<string, { descriptors: any, keypoints: any }> = new Map();
    const orb = new cv.ORB();
    const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);

    try {
      for (const record of transferableImageRecords) {
        let img = new cv.Mat();
        let gray = new cv.Mat();
        let keypoints = new cv.KeyPointVector();
        let descriptors = new cv.Mat();

        try {
          img = cv.imdecode(new cv.Mat(1, record.data.byteLength, cv.CV_8U, new Uint8Array(record.data)), cv.IMREAD_COLOR);
          if (img.empty()) {
            results.push({
              imageId: record.id,
              filename: record.filename,
              brightness: 0, sharpness: 0, isValid: false, messages: ['Failed to decode image.']
            });
            continue;
          }
          cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY, 0);

          orb.detectAndCompute(gray, new cv.Mat(), keypoints, descriptors);

          imageMats.set(record.id, img);
          grayMats.set(record.id, gray);
          descriptorsMap.set(record.id, { descriptors, keypoints });

        } catch (err: any) {
          console.error(`Error processing image ${record.filename} for features:`, err);
          results.push({
            imageId: record.id,
            filename: record.filename,
            brightness: 0, sharpness: 0, isValid: false, messages: [`Error processing image: ${err.message}`]
          });
          img.delete();
          gray.delete();
          keypoints.delete();
          descriptors.delete();
        }
      }

      for (const record of transferableImageRecords) {
        const validationMessages: string[] = [];
        let isValid = true;
        let brightness = 0;
        let sharpness = 0;
        let redundancyScore: number | undefined;
        let redundantWith: string | undefined;

        const gray = grayMats.get(record.id);
        const descData = descriptorsMap.get(record.id);

        if (!gray || !descData) {
          const existingResult = results.find(r => r.imageId === record.id);
          if (existingResult) continue;
          results.push({
            imageId: record.id,
            filename: record.filename,
            brightness: 0, sharpness: 0, isValid: false, messages: ['Image data or features not available.']
          });
          continue;
        }

        const mean = cv.mean(gray);
        brightness = mean[0];
        if (brightness < this.BRIGHTNESS_MIN_THRESHOLD || brightness > this.BRIGHTNESS_MAX_THRESHOLD) {
          validationMessages.push(`Brightness (${brightness.toFixed(2)}) is outside optimal range (${this.BRIGHTNESS_MIN_THRESHOLD}-${this.BRIGHTNESS_MAX_THRESHOLD}).`);
          isValid = false;
        }

        let laplacian = new cv.Mat();
        let meanMat = new cv.Mat();
        let stdDevMat = new cv.Mat();
        try {
          cv.Laplacian(gray, laplacian, cv.CV_64F);
          cv.meanStdDev(laplacian, meanMat, stdDevMat);
          sharpness = stdDevMat.data64F[0] ** 2;
          if (sharpness < this.SHARPNESS_THRESHOLD) {
            validationMessages.push(`Sharpness (${sharpness.toFixed(2)}) is low. Image might be blurry (threshold: ${this.SHARPNESS_THRESHOLD}).`);
            isValid = false;
          }
        } catch (err: any) {
          console.error(`Error calculating sharpness for ${record.filename}:`, err);
          validationMessages.push(`Error calculating sharpness: ${err.message}`);
          isValid = false;
        } finally {
          laplacian.delete();
          meanMat.delete();
          stdDevMat.delete();
        }

        const currentDescriptors = descData.descriptors;
        const currentKeypoints = descData.keypoints;

        if (currentDescriptors.empty()) {
          validationMessages.push('No features found for redundancy check.');
        } else {
          let maxMatchRatio = 0;
          let bestMatchId: string | undefined;

          for (const otherRecord of transferableImageRecords) {
            if (record.id === otherRecord.id) continue;

            const otherDescData = descriptorsMap.get(otherRecord.id);
            if (!otherDescData || otherDescData.descriptors.empty()) continue;

            let matches = new cv.DMatchVector();
            try {
              bf.match(currentDescriptors, otherDescData.descriptors, matches);
              const numMatches = matches.size();
              const minFeatures = Math.min(currentKeypoints.size(), otherDescData.keypoints.size());

              if (minFeatures > 0) {
                const matchRatio = numMatches / minFeatures;
                if (matchRatio > maxMatchRatio) {
                  maxMatchRatio = matchRatio;
                  bestMatchId = otherRecord.id;
                }
              }
            } catch (err: any) {
              console.warn(`Error comparing features between ${record.filename} and ${otherRecord.filename}:`, err);
            } finally {
              matches.delete();
            }
          }

          redundancyScore = maxMatchRatio;
          if (maxMatchRatio > this.REDUNDANCY_MATCH_RATIO_THRESHOLD) {
            validationMessages.push(`Image is highly redundant with image ID ${bestMatchId} (match ratio: ${(maxMatchRatio * 100).toFixed(1)}%, threshold: ${(this.REDUNDANCY_MATCH_RATIO_THRESHOLD * 100).toFixed(1)}%).`);
            redundantWith = bestMatchId;
            isValid = false;
          }
        }

        results.push({
          imageId: record.id,
          filename: record.filename,
          brightness,
          sharpness,
          redundancyScore,
          redundantWith,
          isValid,
          messages: validationMessages,
        });
      }
    } catch (error) {
      console.error('Error during batch photo validation:', error);
      throw error;
    } finally {
      orb.delete();
      bf.delete();
      imageMats.forEach(mat => mat.delete());
      grayMats.forEach(mat => mat.delete());
      descriptorsMap.forEach(data => {
        data.descriptors.delete();
        data.keypoints.delete();
      });
    }
    console.log('Batch photo validation complete.');
    return results;
  }
}

Comlink.expose(new PoseEstimationWorker());