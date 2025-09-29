# Distributed Neural Radiance Field (NeRF) System

## Project Overview

This project aims to develop a cutting-edge web-based, distributed Neural Radiance Field (NeRF) system. Its primary objective is to reconstruct intricate 3D scenes from a collection of 2D images. The system leverages federated learning for collaborative model training across multiple user devices and provides real-time 3D scene rendering directly within the browser. This innovative approach democratizes 3D reconstruction, allowing users to contribute to and experience shared virtual environments.

**Key Stakeholders:**
*   **Project Team:** Developers and engineers responsible for building and maintaining the system.
*   **Users/Contributors:** Individuals who capture and upload images, contributing to the collective NeRF model, and interact with the reconstructed scenes.
*   **Researchers:** Academics and industry professionals interested in distributed AI, 3D reconstruction, and real-time graphics in web environments.

## Features

The Distributed NeRF System encompasses a wide array of features designed for robust 3D reconstruction, collaborative learning, and interactive visualization:

### NeRF Core Model & Training
*   **NeRF Architecture:** Implements an 8-layer Multi-Layer Perceptron (MLP) with positional encoding for efficient scene representation.
*   **Vectorized Volume Rendering:** Utilizes `tf.exp()` and `tf.cumsum()` for differentiable alpha composition, enabling efficient volume rendering within the TensorFlow.js graph.
*   **NeRF Training Loop:** Configurable training loop with batching, learning rate decay, and aggressive GPU memory management (`tf.tidy`, `tf.dispose`).
*   **Real Image Data Ingestion:** Robust pipeline for loading real image data, extracting camera parameters, and generating rays for training.
*   **Adaptive Ray Sampling:** Optimizes volume rendering by performing a coarse pass to identify high-density regions, followed by fine sampling in those areas.
*   **Quantized Weights:** Reduces model size and accelerates inference by quantizing NeRF model weights to 8-bit integers, with corresponding GLSL shader updates.

### Camera Pose Estimation & Scene Reconstruction
*   **Structure from Motion (SfM):** WebGPU compute shaders for accelerated FAST-like keypoint detection and simplified Hamming-like descriptor matching, with OpenCV.js fallback for robust descriptor computation (ORB) and essential matrix estimation (RANSAC).
*   **Global Camera Pose Refinement:** Uses `gl-matrix` to combine relative poses into a consistent global world coordinate system.
*   **Device Sensor Integration:** Incorporates `DeviceOrientationEvent` data to provide initial camera pose estimates.
*   **Full OpenCV.js Library Integration:** Seamless integration of `opencv.js` for advanced computer vision tasks.
*   **Auto-Exposure and Focus Control:** Leverages MediaDevices API for continuous auto-exposure and focus during image capture.
*   **Real-Time Pose Feedback:** WebGPU-accelerated keypoint detection and matching for real-time camera pose visualization in a React overlay.
*   **Coverage Meter:** A UI component to guide users in capturing images from diverse angles for optimal scene coverage.
*   **Batch Photo Validation:** Preprocesses captured/uploaded photos using WebGPU/OpenCV.js to check brightness, sharpness, and redundancy, filtering out low-quality inputs.

### Federated Learning & Synchronization
*   **P2P Model Synchronization:** Utilizes PeerJS for direct peer-to-peer communication and MessagePack for efficient serialization of model updates.
*   **Federated Averaging (FedAvg):** Aggregates delta weights (ΔW) from multiple peers to update the global model.
*   **Differential Privacy:** Adds Gaussian noise to shared model weights for enhanced privacy.
*   **Basic Byzantine Detection:** Implements checks to reject potentially malicious or corrupted weight updates.
*   **Federated Learning Communication Schedule:** Coordinates local training iterations and aggregation intervals across peers.
*   **Optimized Delta Sharing:** Transmits only the difference (delta) between local and global model weights, significantly reducing P2P bandwidth.

### Real-Time Rendering & Optimization
*   **Three.js Scene & Ray Marching Renderer:** Establishes a Three.js environment with a custom `THREE.ShaderMaterial` for GLSL-based ray marching.
*   **Custom GLSL Shader for NeRF Inference:** Implements ray marching and volume accumulation directly in GLSL for high-performance inference.
*   **Performance Optimization:** Employs Comlink Web Workers to offload TensorFlow.js operations, with future plans for WGSL compute shaders and progressive rendering (low-res during movement, high-res when static).
*   **NeRF-Voxel Grid Caching:** Generates a 3D voxel grid of NeRF properties for faster, GPU-accelerated rendering using `THREE.DataTexture3D`.
*   **Centralized NeRF Rendering Pipeline:** The entire NeRF rendering process (ray generation, sampling, inference, volume rendering) is encapsulated and executed within a Web Worker for optimal main thread performance.

### User Interface & Data Management
*   **React UI with @react-three/fiber:** Responsive and interactive user interface built with React, integrating `@react-three/fiber` for 3D canvas management.
*   **Fast, Non-blocking Photo Gallery:** Manages user-contributed photos using Dexie.js (IndexedDB) for efficient, non-blocking data access.
*   **Model and Training Data Storage:** Persists NeRF models, training data, and camera poses using Dexie.js, including `modelVersion` tracking for federated learning.
*   **Quality Metrics Calculation & Visualization:** Calculates PSNR and SSIM on validation sets, with plans for visualization using Recharts.
*   **User-friendly Photo Upload:** Supports drag-and-drop, file input, direct camera access, and image paste functionality for easy image ingestion.

### Cross-Cutting Concerns
*   **Coordinate Transformation Management:** Ensures consistent coordinate systems across OpenCV, DeviceOrientation, gl-matrix, and Three.js.
*   **Web Worker Management:** All TensorFlow.js operations are executed within Web Workers, with Comlink handling efficient `ArrayBuffer` transfers.
*   **Aggressive GPU Memory Management:** Extensive use of `tf.tidy()` and `tf.dispose()` to prevent GPU memory exhaustion.

## Architecture Summary

### Overview
The Distributed Neural Radiance Field (NeRF) System is a browser-centric, peer-to-peer application designed for collaborative 3D scene reconstruction. It leverages Web Workers extensively to offload computationally intensive tasks (NeRF training, pose estimation, federated aggregation) from the main UI thread, ensuring a responsive user experience. TensorFlow.js powers the NeRF model and its training, while PeerJS facilitates direct peer-to-peer communication for federated learning. Comlink manages efficient data transfer between the main thread and Web Workers. Three.js and custom GLSL shaders are used for real-time, interactive 3D rendering. All persistent data, including images, camera poses, and NeRF model checkpoints, is stored locally using Dexie.js (an IndexedDB wrapper). The system operates without a central server for core NeRF processing or federated learning, relying on the distributed nature of connected browsers. This updated version includes optimized federated learning with delta sharing, voxel grid caching for rendering, a centralized rendering pipeline within a Web Worker for enhanced performance, and comprehensive user image ingestion capabilities including upload, direct camera capture, and image pasting.

### Key Components
*   **MainApplicationUI (UI-001):** The main React application shell, orchestrating UI components and Web Worker interactions. It handles user image ingestion and integrates real-time pose feedback.
*   **PhotoGallery (UI-002):** A React component for displaying user-contributed photos, interacting with `NeRFDatabaseService` for persistence and `ImageIngestionUI` for data.
*   **NeRFRenderer (RENDER-001):** Manages real-time 3D visualization using Three.js. It displays rendered images from `NeRFModelWorker` and integrates `THREE.DataTexture3D` for voxel grid caching.
*   **NeRFModelWorker (WORKER-001):** A Web Worker for all TensorFlow.js NeRF operations, including model initialization, training, inference, voxel grid generation, and the centralized rendering pipeline. Ensures aggressive GPU memory management.
*   **PoseEstimationWorker (WORKER-002):** A Web Worker for camera pose estimation, utilizing WebGPU (with OpenCV.js fallback) for SfM, `gl-matrix` for global pose refinement, and device sensor integration. Also handles auto-exposure, focus, real-time pose feedback, and batch photo validation.
*   **FederatedTrainerWorker (WORKER-003):** A Web Worker managing federated learning, including PeerJS for P2P sync, MessagePack for serialization, FedAvg, differential privacy, basic Byzantine detection, and optimized delta sharing.
*   **NeRFDatabaseService (DATA-001):** A service layer abstracting IndexedDB interactions via Dexie.js, managing storage of images, camera poses, and NeRF model checkpoints.
*   **CoordinateTransformer (UTIL-001):** A utility module for converting between OpenCV, DeviceOrientation, gl-matrix, and Three.js coordinate systems.
*   **NeRFDataUtils (UTIL-002):** A utility module for NeRF-specific data handling, including ray generation, point sampling, and volume rendering calculations.
*   **ImageIngestionUI (UI-003):** React component for photo uploads, direct camera access, and image pasting, interacting with `NeRFDatabaseService` and `PoseEstimationWorker`.
*   **PoseFeedbackOverlay (UI-004):** React component for overlaying real-time camera pose visualization and a coverage meter during image capture.

## How to Set Up & Run

This guide will walk you through setting up and running the Distributed NeRF System locally.

### Prerequisites
Before you begin, ensure you have the following installed:
*   **Node.js:** Version 18 or higher. You can download it from [nodejs.org](https://nodejs.org/).
*   **npm (Node Package Manager):** Usually comes with Node.js. You can check its version with `npm -v`.
*   **Git:** For cloning the repository.
*   **PeerJS Server:** A simple PeerJS server is required for peer-to-peer communication. You can install it globally:
    ```bash
    npm install -g peerjs
    ```

### 1. Clone the Repository
First, clone the project repository to your local machine:
```bash
git clone <repository-url>
cd distributed-nerf-system
```

### 2. Install Dependencies
Install the necessary Node.js packages:
```bash
npm install
# or
yarn install
```

### 3. OpenCV.js Setup
The `PoseEstimationWorker` relies on `opencv.js`. You need to manually download the `opencv.js` file and place it in your project's `public/` directory.

1.  **Download `opencv.js`:** Visit the [OpenCV.js releases page](https://docs.opencv.org/4.x/d5/d10/tutorial_js_setup.html) or search for a CDN link. Download the `opencv.js` file (e.g., `opencv.js` or `opencv.wasm.js`).
2.  **Place in `public/`:** Create a `public/` directory in the root of your project if it doesn't exist, and place the downloaded `opencv.js` file inside it.

### 4. Start the PeerJS Server
Open a new terminal window and start the PeerJS server. The application is configured to connect to `localhost:9000` with the path `/myapp`.
```bash
peerjs --port 9000 --path /myapp
```
Keep this terminal running as long as you are using the federated learning features.

### 5. Run the Application
In your project's root directory, start the React development server:
```bash
npm start
# or
yarn start
```
This will open the application in your default web browser, usually at `http://localhost:3000/`.

### 6. Basic Usage
To experience the federated learning and rendering features:

1.  **Open Multiple Tabs:** Open several browser tabs (e.g., 2-3) and navigate to `http://localhost:3000/` in each. Each tab will represent a different peer in the federated network.
2.  **Start Federated Session:** In each tab, locate the "Federated Learning" section in the left control panel.
    *   Enter a **Shared Secret** (e.g., `mysecretkey`). This secret is used to authenticate peer connections. All peers must use the same secret.
    *   Click the "Start Federated Session" button. Each tab will display its unique Peer ID.
3.  **Connect Peers:** In one tab, copy its Peer ID. In another tab, paste this ID into the "Target Peer ID" input field and click "Connect". Repeat this to connect all your open tabs to each other.
4.  **Perform Local Training:** Click the "Perform Local Training Step" button in any tab. This will simulate a local training step using dummy ray data. Observe the "Local Training Epoch" increment.
    *   The system is configured to send model updates (deltas) to connected peers every 5 local training iterations.
    *   Federated aggregation will automatically occur every 10 seconds if enough peers (minimum 1, configurable) have sent their updates.
5.  **Generate Voxel Grid & Toggle Rendering:**
    *   Click "Generate Voxel Grid" to compute a 3D voxel representation of the NeRF scene. This will offload computation to the `NeRFModelWorker`.
    *   Click "Switch to Voxel Grid Rendering" to toggle between the default ray-marching renderer and the GPU-accelerated voxel grid renderer. You can adjust the "Max Ray Marching Steps" for the voxel grid rendering to control quality vs. performance.

Enjoy exploring the distributed NeRF system!
