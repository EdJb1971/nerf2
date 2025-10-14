# Architecture Document: Distributed NeRF System

## 1. Overview

This document outlines the architecture of the Distributed NeRF System, a web-based application for 3D scene reconstruction using Neural Radiance Fields (NeRF). The system is designed to be decentralized, leveraging federated learning for collaborative model training across multiple clients and performing all heavy computations (training, rendering, pose estimation) in the browser.

The architecture is centered around a main React-based UI thread and several Web Workers that handle computationally intensive tasks, ensuring the UI remains responsive.

## 2. High-Level Architecture

The system is composed of three main layers:

1.  **UI Layer (React)**: The user-facing interface, built with React and TypeScript. It manages user interactions, orchestrates the workers, and displays the final rendered 3D scene.
2.  **Worker Layer**: Offloads heavy computations from the main thread. This layer consists of three specialized web workers:
    *   `NeRFModelWorker`: For all NeRF-related operations using TensorFlow.js.
    *   `PoseEstimationWorker`: For camera pose estimation using OpenCV.js.
    *   `FederatedTrainerWorker`: For managing the peer-to-peer federated learning process.
3.  **Data Layer**: Handles local data persistence using IndexedDB (via Dexie.js) to store images, camera poses, and model checkpoints.

## 3. Key Components

### 3.1. Frontend (UI Layer)

-   **`App.tsx`**: The root React component. It serves as the main application controller, initializing workers, managing application state, and handling user events. It now includes:
    -   An automated pose estimation workflow that can be triggered by the user to run a sequential SfM pipeline on all uploaded images.
    -   A progress indicator for the pose estimation process.
    -   Camera intrinsics estimation using EXIF data from images (with a fallback to a heuristic), removing previous hardcoded values.
    -   A corrected NeRF training loop that uses random sampling of images at each iteration, which is a more effective training strategy.
-   **`components/NeRFRenderer.tsx`**: A Three.js-based component (using `@react-three/fiber`) for rendering the NeRF scene. It supports two rendering modes:
    1.  **Progressive Ray Marching**: Communicates with the `NeRFModelWorker` to render the scene. It requests low-resolution renders during camera movement and a high-resolution render when idle.
    2.  **Voxel Grid Rendering**: Uses a pre-computed 3D voxel grid and a custom GLSL shader to render the scene on the GPU. The bounding box for this grid is now adaptively calculated based on the estimated camera poses, instead of being hardcoded.
-   **`components/PhotoGallery.tsx`**: Displays a gallery of user-contributed images stored in the local database.
-   **Image Ingestion**: The UI allows users to add images via file upload, drag-and-drop, camera capture, and pasting from the clipboard.

### 3.2. Web Workers (Worker Layer)

The use of Web Workers is a cornerstone of this architecture, preventing the browser from freezing during intensive computations. `Comlink` is used to provide a clean, promise-based interface to the workers.

-   **`workers/NeRFModelWorker.ts`**:
    -   **Technology**: TensorFlow.js.
    -   **Responsibilities**:
        -   Defines and initializes the NeRF model (an 8-layer MLP).
        -   Provides methods for `train`ing the model on a batch of rays and `predict`ing the color and density for a set of 3D points.
        -   Contains a `renderNeRFImage` method that encapsulates the entire rendering pipeline (ray generation, sampling, inference, volume rendering).
        -   Can generate a 3D voxel grid (`generateVoxelGrid`) for accelerated rendering.
        -   Exposes methods to get and set model weights, which is essential for federated learning.

-   **`workers/PoseEstimationWorker.ts`**:
    -   **Technology**: OpenCV.js.
    -   **Responsibilities**:
        -   Estimates relative camera poses between image pairs using Structure from Motion (SfM) techniques (ORB features, feature matching, and Essential Matrix estimation).
        -   Integrates `DeviceOrientationEvent` data for initial pose estimates.
        -   Includes a `validateBatchPhotos` method to assess image quality (brightness, sharpness, and redundancy).
        -   Contains a placeholder for global pose refinement (`refineGlobalPoses`). A warning has been added to the console to inform developers of its limitations.

-   **`workers/FederatedTrainerWorker.ts`**:
    -   **Technology**: PeerJS for WebRTC, `msgpack` for serialization.
    -   **Responsibilities**:
        -   Manages the peer-to-peer network for federated learning.
        -   Implements Federated Averaging (FedAvg). It sends and receives model *deltas* to reduce bandwidth.
        -   Applies differential privacy by adding Gaussian noise to the shared weight deltas.
        -   Includes a basic Byzantine fault tolerance mechanism to validate incoming weights.
        -   Orchestrates the aggregation of weights from peers and updates the local model.

### 3.3. Services and Utilities (Data Layer & Utils)

-   **`services/NeRFDatabaseService.ts`**:
    -   **Technology**: Dexie.js (IndexedDB wrapper).
    -   **Responsibilities**: Provides a simple API to store and retrieve `ImageRecord`, `CameraPoseRecord`, and `ModelCheckpointRecord` from the browser's IndexedDB.

-   **`utils/CoordinateTransformer.ts`**: A crucial utility for converting transformation matrices and vectors between the different coordinate systems used by OpenCV, DeviceOrientation, gl-matrix, and Three.js.

-   **`utils/NeRFDataUtils.ts`**: Contains helper functions for NeRF-specific operations like ray generation (`getRays`), point sampling (`samplePointsAlongRays`), and volume rendering (`volumeRender`).

## 4. Data Flow

### 4.1. Training Data Flow

1.  **Image Upload**: User uploads images, which are stored in IndexedDB via `NeRFDatabaseService`.
2.  **Pose Estimation**: The user can now trigger an automated pose estimation pipeline from the UI. This process:
    -   Estimates camera intrinsics from EXIF data.
    -   Runs a sequential SfM process on all images to calculate their camera poses.
    -   Saves the poses to the database.
3.  **Training Initiation**: The user starts the training process from the UI, which now runs for a configurable number of iterations.
4.  **Data Preparation & Local Training**: For each training iteration, `App.tsx`:
    -   Randomly selects an image and its corresponding pose.
    -   Generates a batch of rays from that image.
    -   Sends the rays to the `NeRFModelWorker` to perform a single training step.
    This random sampling approach is more effective for learning a consistent 3D scene.
5.  **Federated Sync**:
    -   After a few local iterations, `FederatedTrainerWorker` requests the model weights from `NeRFModelWorker`.
    -   It calculates the delta, adds noise, and sends it to connected peers.
    -   It receives deltas from other peers and, on a set interval, aggregates them.
    -   The aggregated delta is applied to the last global model, and the new weights are sent to `NeRFModelWorker` to update the local model.

### 42. Rendering Data Flow

1.  **Camera Movement**: The user moves the camera in the `NeRFRenderer`.
2.  **Render Request**: The renderer component sends the new camera matrix to the `NeRFModelWorker` via `renderNeRFImage`.
3.  **Worker-Side Rendering**: The `NeRFModelWorker` performs the full rendering pipeline (ray generation, sampling, querying the NeRF model, and volume rendering).
4.  **Frame Update**: The worker returns the rendered image as an `ArrayBuffer`, which is then displayed on a texture in the Three.js scene.

## 5. Placeholders, Mock Code, and Incomplete Functionality

-   **Global Pose Refinement**: The `refineGlobalPoses` method in `PoseEstimationWorker` is a placeholder that performs a simple iterative smoothing. It is not a full Bundle Adjustment implementation. A warning has been added to the console to highlight this limitation.
-   **`opencv.js.patch`**: The presence of this patch file indicates a potential workaround for an issue with the `opencv.js` library, which might be brittle.
-   **Error Handling and User Feedback**: While a progress indicator has been added for pose estimation, the application could still benefit from more comprehensive error handling and user feedback for other long-running tasks.
-   **Federation Security**: The federated learning sessions are secured only by a shared secret string. This is a basic implementation and would need to be enhanced for a production environment.
