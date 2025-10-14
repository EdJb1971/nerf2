import React, { useEffect, useState, useCallback, useRef } from 'react';
import * as Comlink from 'comlink';
import * as tf from '@tensorflow/tfjs';
import { NeRFCanvas } from './components/NeRFRenderer';
import { PhotoGallery } from './components/PhotoGallery';
import { NeRFModelWorker } from './workers/NeRFModelWorker';
import { PoseEstimationWorker, PhotoValidationResult } from './workers/PoseEstimationWorker';
import { FederatedTrainerWorker } from './workers/FederatedTrainerWorker';
import { nerfDatabaseService, CameraPoseRecord, ImageRecord } from './services/NeRFDatabaseService';
import { nerfDataUtils } from './utils/NeRFDataUtils';
import exifr from 'exifr';

type NeRFModelWorkerProxy = Comlink.Remote<NeRFModelWorker>;
type PoseEstimationWorkerProxy = Comlink.Remote<PoseEstimationWorker>;
type FederatedTrainerWorkerProxy = Comlink.Remote<FederatedTrainerWorker>;

// New constants for real data training
const TRAINING_IMAGE_RESOLUTION = 128; // Images will be resized to this resolution for training
const TRAINING_BATCH_SIZE = 1024; // Number of rays to sample per training step

const extractCameraIntrinsics = async (imageBlob: Blob): Promise<Float32Array | null> => {
  try {
    const exif = await exifr.parse(imageBlob);
    
    if (exif && exif.ImageWidth && exif.ImageHeight) {
      const width = exif.ImageWidth;
      const height = exif.ImageHeight;
      
      // If focal length is available in EXIF
      let focalLength = 1.2 * Math.max(width, height); // Default heuristic
      
      if (exif.FocalLength && exif.FocalLengthIn35mmFormat) {
        // Convert 35mm equivalent to actual focal length in pixels
        const sensorWidth = 36; // 35mm film width in mm
        const focalLengthMM = exif.FocalLength;
        focalLength = (focalLengthMM / sensorWidth) * width;
      }
      
      return new Float32Array([
        focalLength, 0, width / 2,
        0, focalLength, height / 2,
        0, 0, 1
      ]);
    }
  } catch (err) {
    console.warn('Failed to extract EXIF data:', err);
  }
  
  return null;
};

const calculateCameraIntrinsics = async (image: ImageRecord): Promise<Float32Array> => {
  const fromExif = await extractCameraIntrinsics(image.data);
  if (fromExif) {
    return fromExif;
  }

  // Fallback to heuristic
  const width = TRAINING_IMAGE_RESOLUTION;
  const height = TRAINING_IMAGE_RESOLUTION;
  const focalLength = 1.2 * Math.max(width, height);
  return new Float32Array([
    focalLength, 0, width / 2,
    0, focalLength, height / 2,
    0, 0, 1
  ]);
};

function App() {
  const [nerfModelWorkerProxy, setNeRFModelWorkerProxy] = useState<NeRFModelWorkerProxy | null>(null);
  const [poseEstimationWorkerProxy, setPoseEstimationWorkerProxy] = useState<PoseEstimationWorkerProxy | null>(null);
  const [federatedTrainerWorkerProxy, setFederatedTrainerWorkerProxy] = useState<FederatedTrainerWorkerProxy | null>(null);
  const [cameraPoses, setCameraPoses] = useState<CameraPoseRecord[]>([]);
  const [images, setImages] = useState<ImageRecord[]>([]); // State for images
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFederatedSessionActive, setIsFederatedSessionActive] = useState(false);
  const [peerId, setPeerId] = useState('');
  const [targetPeerId, setTargetPeerId] = useState('');
  const [localTrainingEpoch, setLocalTrainingEpoch] = useState(0);

  const [federationSecret, setFederationSecret] = useState<string>(''); // User must provide

  const [iterationsPerRound, setIterationsPerRound] = useState(5);
  const [aggregationIntervalMs, setAggregationIntervalMs] = useState(10000);
  const [minPeersForAggregation, setMinPeersForAggregation] = useState(1);
  const [totalTrainingIterations, setTotalTrainingIterations] = useState(50000);

  // New states for Voxel Grid Caching
  const [generateVoxelGridTrigger, setGenerateVoxelGridTrigger] = useState(0);
  const [useVoxelGridRendering, setUseVoxelGridRendering] = useState(false);
  const [voxelGridMaxSteps, setVoxelGridMaxSteps] = useState(100);

  // States for Camera Capture
  const [showCameraModal, setShowCameraModal] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null); // For file upload

  // New states for Photo Validation
  const [validationResults, setValidationResults] = useState<PhotoValidationResult[]>([]);
  const [showValidationResults, setShowValidationResults] = useState(false);
  const [poseEstimationProgress, setPoseEstimationProgress] = useState<{
    current: number;
    total: number;
    stage: string;
  } | null>(null);

  const refreshImages = useCallback(async () => {
    try {
      const storedImages = await nerfDatabaseService.getImages();
      setImages(storedImages);
      console.log(`Refreshed: Loaded ${storedImages.length} images.`);
    } catch (err: any) {
      console.error('Failed to refresh images:', err);
      setError(err.message || 'An error occurred while refreshing images.');
    }
  }, []);

  // Handle file upload (STORY-505)
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    console.log(`Uploading ${files.length} files...`);
    setError(null);

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        await nerfDatabaseService.saveImage(file, file.name);
      }
      await refreshImages();
      console.log('Files uploaded successfully.');
    } catch (err: any) {
      console.error('Error uploading files:', err);
      setError(err.message || 'An error occurred during file upload.');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // Clear the input
      }
    }
  }, [refreshImages]);

  // Camera capture functions (STORY-506)
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err: any) {
      console.error('Error accessing camera:', err);
      setError(err.message || 'Failed to access camera.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const takePhoto = useCallback(async () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
          if (blob) {
            try {
              await nerfDatabaseService.saveImage(blob, `camera-capture-${Date.now()}.png`);
              await refreshImages();
              console.log('Photo captured and saved.');
              setShowCameraModal(false); // Close modal after capture
              stopCamera();
            } catch (err: any) {
              console.error('Error saving captured photo:', err);
              setError(err.message || 'Failed to save captured photo.');
            }
          }
        }, 'image/png');
      }
    }
  }, [refreshImages, stopCamera]);

  // Handle image paste (STORY-507)
  useEffect(() => {
    const handlePaste = async (event: ClipboardEvent) => {
      if (!event.clipboardData) return;

      const items = event.clipboardData.items;
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.indexOf('image') !== -1) {
          const blob = item.getAsFile();
          if (blob) {
            try {
              await nerfDatabaseService.saveImage(blob, `pasted-image-${Date.now()}.${blob.type.split('/')[1]}`);
              await refreshImages();
              console.log('Pasted image saved.');
            } catch (err: any) {
              console.error('Error saving pasted image:', err);
              setError(err.message || 'Failed to save pasted image.');
            }
          }
        }
      }
    };

    document.body.addEventListener('paste', handlePaste);
    return () => {
      document.body.removeEventListener('paste', handlePaste);
    };
  }, [refreshImages]);

  useEffect(() => {
    const initWorkers = async () => {
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('TensorFlow.js backend initialized:', tf.getBackend());

        const nerfWorker = new Worker(new URL('./workers/NeRFModelWorker', import.meta.url), { type: 'module' });
        const nerfProxy = Comlink.wrap<NeRFModelWorker>(nerfWorker);
        await nerfProxy.initModel();
        setNeRFModelWorkerProxy(nerfProxy);
        console.log('NeRFModelWorker initialized.');

        const poseWorker = new Worker(new URL('./workers/PoseEstimationWorker', import.meta.url), { type: 'module' });
        const poseProxy = Comlink.wrap<PoseEstimationWorker>(poseWorker);
        setPoseEstimationWorkerProxy(poseProxy);
        console.log('PoseEstimationWorker initialized.');

        const federatedWorker = new Worker(new URL('./workers/FederatedTrainerWorker', import.meta.url), { type: 'module' });
        const federatedProxy = Comlink.wrap<FederatedTrainerWorker>(federatedWorker);
        await federatedProxy.initNeRFModelWorker(nerfProxy);
        await federatedProxy.setCommunicationSchedule({
          iterationsPerRound,
          aggregationIntervalMs,
          minPeersForAggregation
        });
        setFederatedTrainerWorkerProxy(federatedProxy);
        console.log('FederatedTrainerWorker initialized.');

        const storedPoses = await nerfDatabaseService.getCameraPoses();
        setCameraPoses(storedPoses);
        console.log(`Loaded ${storedPoses.length} camera poses.`);

        await refreshImages(); // Load initial images

      } catch (err: any) {
        console.error('Failed to initialize workers or fetch data:', err);
        setError(err.message || 'An unknown error occurred during initialization.');
      } finally {
        setLoading(false);
      }
    };

    initWorkers();

    return () => {
      nerfModelWorkerProxy?.[Comlink.releaseProxy]();
      poseEstimationWorkerProxy?.[Comlink.releaseProxy]();
      federatedTrainerWorkerProxy?.[Comlink.releaseProxy]();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [iterationsPerRound, aggregationIntervalMs, minPeersForAggregation, refreshImages]);

  const startFederatedSession = useCallback(async () => {
    if (federatedTrainerWorkerProxy) {
      try {
        const id = `nerf-peer-${Math.random().toString(36).substring(2, 9)}`;
        setPeerId(id);
        await federatedTrainerWorkerProxy.startSession(id, federationSecret, {
          host: import.meta.env.VITE_PEERJS_HOST || 'localhost',
          port: parseInt(import.meta.env.VITE_PEERJS_PORT || '9000', 10),
          path: import.meta.env.VITE_PEERJS_PATH || '/myapp'
        });
        await federatedTrainerWorkerProxy.startAggregationTimer();
        setIsFederatedSessionActive(true);
        console.log('Federated session started.');
      } catch (err: any) {
        console.error('Failed to start federated session:', err);
        setError(err.message || 'Failed to start federated session.');
      }
    }
  }, [federatedTrainerWorkerProxy, federationSecret]);

  const connectToPeer = useCallback(async () => {
    if (federatedTrainerWorkerProxy && targetPeerId) {
      try {
        await federatedTrainerWorkerProxy.connectToPeer(targetPeerId);
        console.log(`Connected to peer: ${targetPeerId}`);
        setTargetPeerId('');
      } catch (err: any) {
        console.error('Failed to connect to peer:', err);
        setError(err.message || 'Failed to connect to peer.');
      }
    }
  }, [federatedTrainerWorkerProxy, targetPeerId]);

  const startRealDataTraining = useCallback(async () => {
    if (!nerfModelWorkerProxy || !federatedTrainerWorkerProxy) {
      setError('Workers not initialized for real data training.');
      return;
    }
    if (images.length === 0 || cameraPoses.length === 0) {
      setError('No images or camera poses available for training.');
      return;
    }

    console.log(`Starting real data training for ${totalTrainingIterations} iterations...`);
    setError(null);

    try {
      const posesByImageId = new Map(cameraPoses.map(pose => [pose.imageId, pose]));
      const validImages = images.filter(img => posesByImageId.has(img.id));

      if (validImages.length === 0) {
        setError('No images have corresponding camera poses.');
        return;
      }

      for (let iter = 0; iter < totalTrainingIterations; iter++) {
        // Select a random image for this iteration
        const randomImageIndex = Math.floor(Math.random() * validImages.length);
        const imageRecord = validImages[randomImageIndex];
        const poseRecord = posesByImageId.get(imageRecord.id)!;

        try {
          tf.engine().startScope();
          const imgBlob = imageRecord.data;
          const imgBitmap = await createImageBitmap(imgBlob);

          const H = TRAINING_IMAGE_RESOLUTION;
          const W = TRAINING_IMAGE_RESOLUTION;
          const focalX = W;
          const focalY = H;
          const centerX = W / 2;
          const centerY = H / 2;

          const imageTensor = tf.browser.fromPixels(imgBitmap).resizeBilinear([H, W]).div(255.0);
          imgBitmap.close();

          const rays = nerfDataUtils.getRays(
            H, W, focalX, focalY, centerX, centerY, poseRecord.poseMatrix
          );
          const allRayOrigins = rays.rayOrigins;
          const allRayDirections = rays.rayDirections;
          const allTargetRgb = imageTensor.reshape([-1, 3]);

          const numPixels = H * W;
          const indices = tf.randomUniform([TRAINING_BATCH_SIZE], 0, numPixels, 'int32');

          const sampledRayOrigins = tf.gather(allRayOrigins, indices);
          const sampledRayDirections = tf.gather(allRayDirections, indices);
          const sampledTargetRgb = tf.gather(allTargetRgb, indices);

          const loss = await nerfModelWorkerProxy.train(
            Comlink.transfer(sampledRayOrigins, [(await sampledRayOrigins.data()).buffer]),
            Comlink.transfer(sampledRayDirections, [(await sampledRayDirections.data()).buffer]),
            Comlink.transfer(sampledTargetRgb, [(await sampledTargetRgb.data()).buffer])
          );

          const currentMetrics = await nerfModelWorkerProxy.getTrainingMetrics();
          setLocalTrainingEpoch(currentMetrics.epoch);
          console.log(`Training Iteration ${iter + 1}/${totalTrainingIterations}. Image: ${imageRecord.filename}, Epoch: ${currentMetrics.epoch}, Loss: ${loss.toFixed(6)}`);

          const iterationCount = await federatedTrainerWorkerProxy.incrementLocalTrainingIteration();
          const schedule = await federatedTrainerWorkerProxy.getCommunicationSchedule();

          if (iterationCount % schedule.iterationsPerRound === 0) {
            console.log(`Iterations per round (${schedule.iterationsPerRound}) met. Sending weights to peers.`);
            await federatedTrainerWorkerProxy.sendWeightsToPeers();
          }
        } finally {
            tf.engine().endScope();
        }
      }

      console.log(`Real data training completed for ${totalTrainingIterations} iterations.`);
    } catch (err: any) {
      console.error('Error during real data training:', err);
      setError(err.message || 'An error occurred during real data training.');
    }
  }, [nerfModelWorkerProxy, federatedTrainerWorkerProxy, images, cameraPoses, totalTrainingIterations, iterationsPerRound, aggregationIntervalMs, minPeersForAggregation]);

  const handleValidatePhotos = useCallback(async () => {
    if (!poseEstimationWorkerProxy) {
      setError('Pose Estimation Worker not initialized.');
      return;
    }
    if (images.length === 0) {
      setError('No images available for validation.');
      return;
    }

    console.log(`Starting batch photo validation for ${images.length} images...`);
    setError(null);
    setValidationResults([]); // Clear previous results

    try {
      // Prepare transferable image data: extract ArrayBuffer from each Blob
      const transferableImages = await Promise.all(images.map(async (img) => ({
        id: img.id,
        filename: img.filename,
        mimeType: img.mimeType,
        data: await img.data.arrayBuffer(), // Convert Blob to ArrayBuffer
      })));

      // Collect transferable objects for Comlink
      const transferables: Transferable[] = transferableImages.map(img => img.data);

      const results = await poseEstimationWorkerProxy.validateBatchPhotos(
        Comlink.transfer(transferableImages, transferables)
      );
      setValidationResults(results);
      setShowValidationResults(true);
      console.log('Batch photo validation completed:', results);
    } catch (err: any) {
      console.error('Error during batch photo validation:', err);
      setError(err.message || 'An error occurred during photo validation.');
    }
  }, [poseEstimationWorkerProxy, images]);

  const handleGenerateVoxelGrid = useCallback(() => {
    setGenerateVoxelGridTrigger(prev => prev + 1);
  }, []);

  const handleToggleVoxelGridRendering = useCallback(() => {
    setUseVoxelGridRendering(prev => !prev);
  }, []);

  const handleEstimateAllPoses = useCallback(async () => {
    if (!poseEstimationWorkerProxy) {
      setError('Pose Estimation Worker not initialized.');
      return;
    }
    if (images.length < 2) {
      setError('Need at least 2 images for pose estimation.');
      return;
    }

    console.log(`Starting SfM pipeline for ${images.length} images...`);
    setError(null);
    setPoseEstimationProgress({ current: 0, total: images.length, stage: 'Starting...' });

    try {
      // Step 1: Set first image as reference (identity pose)
      const referenceImage = images[0];
      const identityPose = new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ]);
      await nerfDatabaseService.saveCameraPose(referenceImage.id, identityPose);
      console.log(`Set ${referenceImage.filename} as reference (identity pose)`);
      
      // Step 2: Estimate relative poses for all other images
      const cameraMatrix = await calculateCameraIntrinsics(referenceImage); // New helper function
      const estimatedPoses: CameraPoseRecord[] = [];
      
      for (let i = 1; i < images.length; i++) {
        setPoseEstimationProgress({ 
          current: i, 
          total: images.length, 
          stage: `Estimating pose ${i}/${images.length - 1}` 
        });
        
        try {
          const result = await poseEstimationWorkerProxy.estimateRelativePoses(
            referenceImage.id,
            images[i].id,
            cameraMatrix,
            undefined // No distortion coefficients
          );
          estimatedPoses.push({
            id: '', // id will be generated by dexie
            imageId: result.image2Id,
            poseMatrix: result.poseMatrix,
            timestamp: Date.now()
          });
          console.log(`Estimated pose for ${images[i].filename}`);
        } catch (err: any) {
          console.warn(`Failed to estimate pose for ${images[i].filename}:`, err.message);
        }
      }
      
      // Step 3: Refine global poses (optional but recommended)
      if (estimatedPoses.length >= 2) {
        setPoseEstimationProgress({ 
          current: images.length, 
          total: images.length, 
          stage: 'Refining global poses...' 
        });
        
        const allPoses = [
          { imageId: referenceImage.id, poseMatrix: identityPose },
          ...estimatedPoses
        ];
        
        const refinedPoses = await poseEstimationWorkerProxy.refineGlobalPoses(allPoses);
        
        // Save refined poses
        for (const pose of refinedPoses) {
          await nerfDatabaseService.saveCameraPose(pose.imageId, pose.poseMatrix);
        }
      }
      
      // Refresh camera poses in UI
      const allPoses = await nerfDatabaseService.getCameraPoses();
      setCameraPoses(allPoses);
      
      console.log(`Pose estimation complete: ${estimatedPoses.length + 1}/${images.length} successful`);
      setError(null);
      
    } catch (err: any) {
      console.error('Error during automated pose estimation:', err);
      setError(err.message || 'Failed to estimate poses');
    } finally {
      setPoseEstimationProgress(null);
    }
  }, [poseEstimationWorkerProxy, images]);

  // UI for camera capture modal
  const CameraCaptureModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-xl max-w-2xl w-full">
        <h3 className="text-xl font-semibold mb-4">Capture Photo</h3>
        <video ref={videoRef} className="w-full h-auto bg-gray-800 mb-4" autoPlay playsInline muted></video>
        <canvas ref={canvasRef} className="hidden"></canvas>
        <div className="flex justify-between">
          <button
            className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
            onClick={takePhoto}
          >
            Take Photo
          </button>
          <button
            className="bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded"
            onClick={() => { setShowCameraModal(false); stopCamera(); }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );

  if (loading) {
    return <div className="flex items-center justify-center h-screen text-lg text-gray-700">Loading application...</div>;
  }

  if (error) {
    return <div className="flex items-center justify-center h-screen text-lg text-red-600">Error: {error}</div>;
  }

  return (
    <div className="App h-screen w-screen flex flex-col">
      <header className="bg-gray-800 text-white p-4 shadow-md">
        <h1 className="text-2xl font-bold">Distributed NeRF System</h1>
      </header>
      <main className="flex-grow flex">
        <div className="w-1/4 bg-gray-100 p-4 overflow-y-auto">
          <h2 className="text-xl font-semibold mb-4">Controls & Gallery</h2>

          <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-medium mb-2">Federated Learning</h3>
            {!isFederatedSessionActive ? (
              <>
                <label className="block text-sm font-medium text-gray-700 mb-1">Shared Secret:</label>
                <input
                  type="password"  
                  placeholder="Federation Shared Secret"
                  className="flex-grow p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 mb-2 w-full"
                  value={federationSecret}
                  onChange={(e) => setFederationSecret(e.target.value)}
                />
                <button
                  className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded w-full mb-2"
                  onClick={startFederatedSession}
                  disabled={!federatedTrainerWorkerProxy || !federationSecret}
                >
                  Start Federated Session
                </button>
              </>
            ) : (
              <>
                <p className="text-sm text-gray-700 mb-2">Your Peer ID: <span className="font-mono text-purple-800">{peerId}</span></p>
                <p className="text-sm text-gray-700 mb-2">Shared Secret: <span className="font-mono text-purple-800">{federationSecret}</span></p>
                <div className="flex mb-2">
                  <input
                    type="text"
                    placeholder="Target Peer ID"
                    className="flex-grow p-2 border rounded-l-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                    value={targetPeerId}
                    onChange={(e) => setTargetPeerId(e.target.value)}
                  />
                  <button
                    className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-r-md"
                    onClick={connectToPeer}
                    disabled={!federatedTrainerWorkerProxy || !targetPeerId}
                  >
                    Connect
                  </button>
                </div>
                <p className="text-sm text-gray-700">Local Training Epoch: {localTrainingEpoch}</p>
              </>
            )}
            <div className="mt-4 pt-4 border-t border-gray-200">
              <h4 className="text-md font-medium mb-2">Federation Schedule (Configurable)</h4>
              <label className="block text-sm font-medium text-gray-700">Iterations per Round:</label>
              <input
                type="number"
                className="p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 mb-2 w-full"
                value={iterationsPerRound}
                onChange={(e) => setIterationsPerRound(parseInt(e.target.value) || 1)}
                min="1"
              />
              <label className="block text-sm font-medium text-gray-700">Aggregation Interval (ms):</label>
              <input
                type="number"
                className="p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 mb-2 w-full"
                value={aggregationIntervalMs}
                onChange={(e) => setAggregationIntervalMs(parseInt(e.target.value) || 1000)}
                min="1000"
              />
              <label className="block text-sm font-medium text-gray-700">Min Peers for Aggregation:</label>
              <input
                type="number"
                className="p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 mb-2 w-full"
                value={minPeersForAggregation}
                onChange={(e) => setMinPeersForAggregation(parseInt(e.target.value) || 1)}
                min="1"
              />
            </div>
          </div>

          <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-medium mb-2">NeRF Training</h3>
            <label className="block text-sm font-medium text-gray-700">Total Training Iterations:</label>
            <input
              type="number"
              className="p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-yellow-500 mb-2 w-full"
              value={totalTrainingIterations}
              onChange={(e) => setTotalTrainingIterations(parseInt(e.target.value) || 1)}
              min="1"
            />
            <button
              className="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-2 px-4 rounded w-full mb-2"
              onClick={startRealDataTraining}
              disabled={!nerfModelWorkerProxy || images.length === 0 || cameraPoses.length === 0}
            >
              Start Training with Real Data ({images.length} images)
            </button>
          </div>

          <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-medium mb-2">Image Ingestion & Validation</h3>
            <input
              type="file"
              ref={fileInputRef}
              multiple
              accept="image/*"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
            />
            <button
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full mb-2"
              onClick={() => fileInputRef.current?.click()}
            >
              Upload Images
            </button>
            <button
              className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded w-full mb-2"
              onClick={() => { setShowCameraModal(true); startCamera(); }}
            >
              Capture from Camera
            </button>
            <button
              className="bg-teal-500 hover:bg-teal-700 text-white font-bold py-2 px-4 rounded w-full mb-2"
              onClick={handleValidatePhotos}
              disabled={!poseEstimationWorkerProxy || images.length === 0}
            >
              Validate Photos ({images.length} images)
            </button>
            <button
              className="bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded w-full mb-2"
              onClick={handleEstimateAllPoses}
              disabled={!poseEstimationWorkerProxy || images.length < 2}
            >
              Estimate All Poses ({images.length} images)
            </button>
            {poseEstimationProgress && (
              <div className="mt-2 p-2 bg-blue-50 rounded">
                <p className="text-sm text-gray-700">{poseEstimationProgress.stage}</p>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all"
                    style={{ width: `${(poseEstimationProgress.current / poseEstimationProgress.total) * 100}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {showValidationResults && ( // Conditionally render validation results
            <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm">
              <h3 className="text-lg font-medium mb-2">Validation Results</h3>
              {validationResults.length === 0 ? (
                <p className="text-gray-600">No results to display.</p>
              ) : (
                <ul className="list-disc list-inside text-sm text-gray-700">
                  {validationResults.map((result) => (
                    <li key={result.imageId} className={`mb-1 ${result.isValid ? 'text-green-700' : 'text-red-700'}`}>
                      <strong>{result.filename}</strong> ({result.isValid ? 'Valid' : 'Invalid'})
                      <ul className="list-disc list-inside ml-4">
                        <li>Brightness: {result.brightness.toFixed(2)}</li>
                        <li>Sharpness: {result.sharpness.toFixed(2)}</li>
                        {result.redundancyScore !== undefined && <li>Redundancy Score: {(result.redundancyScore * 100).toFixed(1)}% {result.redundantWith && `(with ${result.redundantWith})`}</li>}
                        {result.messages.map((msg, idx) => (
                          <li key={idx} className="text-red-500">{msg}</li>
                        ))}
                      </ul>
                    </li>
                  ))}
                </ul>
              )}
              <button
                className="bg-gray-500 hover:bg-gray-600 text-white font-bold py-1 px-3 rounded mt-2 text-sm"
                onClick={() => setShowValidationResults(false)}
              >
                Hide Results
              </button>
            </div>
          )}

          <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-medium mb-2">NeRF Rendering Options</h3>
            <button
              className="bg-indigo-500 hover:bg-indigo-600 text-white font-bold py-2 px-4 rounded w-full mb-2"
              onClick={handleGenerateVoxelGrid}
              disabled={!nerfModelWorkerProxy}
            >
              Generate Voxel Grid
            </button>
            <button
              className={`font-bold py-2 px-4 rounded w-full ${useVoxelGridRendering ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'} text-white`}
              onClick={handleToggleVoxelGridRendering}
              disabled={!nerfModelWorkerProxy}
            >
              {useVoxelGridRendering ? 'Switch to Ray Marching' : 'Switch to Voxel Grid Rendering'}
            </button>
            <div className="mt-4 pt-4 border-t border-gray-200">
              <h4 className="text-md font-medium mb-2">Voxel Grid Settings</h4>
              <label className="block text-sm font-medium text-gray-700">Max Ray Marching Steps:</label>
              <input
                type="number"
                className="p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-2 w-full"
                value={voxelGridMaxSteps}
                onChange={(e) => setVoxelGridMaxSteps(parseInt(e.target.value) || 1)}
                min="1"
                max="500" 
              />
            </div>
          </div>

          <div className="mb-6 p-4 border rounded-lg bg-white shadow-sm">
            <h3 className="text-lg font-medium mb-2">Photo Gallery</h3>
            <PhotoGallery images={images} onImageSelect={(id) => console.log('Image selected:', id)} />
          </div>

          <p className="mt-4 text-sm text-gray-600">
            Currently displaying {cameraPoses.length} estimated camera poses in the viewer.
          </p>
        </div>
        <div className="w-3/4 bg-gray-200 relative">
          {nerfModelWorkerProxy ? (
            <NeRFCanvas
              nerfModelWorkerProxy={nerfModelWorkerProxy}
              cameraPoses={cameraPoses}
              width={window.innerWidth * 0.75}
              height={window.innerHeight - 64}
              generateVoxelGridTrigger={generateVoxelGridTrigger}
              useVoxelGridRendering={useVoxelGridRendering}
              voxelGridMaxSteps={voxelGridMaxSteps}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-lg text-gray-500">
              Waiting for NeRF model worker to initialize...
            </div>
          )}
        </div>
      </main>
      {showCameraModal && <CameraCaptureModal />} 
    </div>
  );
}

export default App;
