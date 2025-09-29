import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree, extend } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import * as Comlink from 'comlink';
import * as tf from '@tensorflow/tfjs';
import { mat4 } from 'gl-matrix';

import { NeRFModelWorker } from '../workers/NeRFModelWorker';
import { nerfDataUtils } from '../utils/NeRFDataUtils';
import { CameraPoseRecord } from '../services/NeRFDatabaseService';

type NeRFModelWorkerProxy = Comlink.Remote<NeRFModelWorker>;

interface NeRFRendererProps {
  nerfModelWorkerProxy: NeRFModelWorkerProxy | null;
  cameraPoses: CameraPoseRecord[];
  width: number;
  height: number;
  focalX?: number;
  focalY?: number;
  centerX?: number;
  centerY?: number;
  generateVoxelGridTrigger: number; // Trigger to generate voxel grid
  useVoxelGridRendering: boolean; // Toggle between ray marching and voxel grid rendering
  voxelGridMaxSteps: number; // New prop for configurable max steps
}

const LOW_RES_WIDTH = 64;
const LOW_RES_HEIGHT = 64;
const HIGH_RES_WIDTH = 256;
const HIGH_RES_HEIGHT = 256;

const N_SAMPLES = 64;
const NEAR_PLANE = 0.0;
const FAR_PLANE = 1.0;
const PERTURB_SAMPLES = true;

// Extend THREE with ShaderMaterial for use in JSX
extend({ ShaderMaterial: THREE.ShaderMaterial });

// Vertex shader for the VoxelGridVisualizer
const voxelVertexShader = `
  varying vec3 vOrigin;
  varying vec3 vDirection;
  void main() {
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      // Calculate ray origin and direction in model space
      vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
      vDirection = normalize(position - vOrigin);
  }
`;

// Fragment shader for the VoxelGridVisualizer (simplified ray marching)
const voxelFragmentShader = `
  uniform sampler3D uVoxelGrid;
  uniform vec3 uGridMin;
  uniform vec3 uGridMax;
  uniform int uMaxSteps;
  varying vec3 vOrigin;
  varying vec3 vDirection;

  // Function to intersect ray with AABB
  vec2 intersectBox(vec3 rayOrigin, vec3 rayDirection, vec3 boxMin, vec3 boxMax) {
      vec3 invR = 1.0 / rayDirection;
      vec3 tbot = invR * (boxMin - rayOrigin);
      vec3 ttop = invR * (boxMax - rayOrigin);
      vec3 tmin = min(ttop, tbot);
      vec3 tmax = max(ttop, tbot);
      vec2 t = vec2(max(tmin.x, max(tmin.y, tmin.z)),
                    min(tmax.x, min(tmax.y, tmax.z)));
      return t;
  }

  void main() {
      vec2 t_hit = intersectBox(vOrigin, vDirection, uGridMin, uGridMax);
      if (t_hit.x > t_hit.y) {
          discard; // Ray misses or starts outside
      }
      float t_start = max(t_hit.x, 0.0); // Start from 0 or box entry
      float t_end = t_hit.y;

      vec3 currentPosition = vOrigin + vDirection * t_start;
      vec3 stepDir = normalize(vDirection);
      float stepSize = 0.01; // Small step size for ray marching
      vec3 step = stepDir * stepSize;

      vec3 accumulatedColor = vec3(0.0);
      float accumulatedAlpha = 0.0;

      for (int i = 0; i < uMaxSteps; ++i) {
          if (accumulatedAlpha > 0.95 || length(currentPosition - vOrigin) > t_end) {
              break;
          }

          // Normalize currentPosition to [0,1] texture coordinates
          vec3 texCoord = (currentPosition - uGridMin) / (uGridMax - uGridMin);
          // Ensure texture coordinates are within [0,1]
          texCoord = clamp(texCoord, 0.0, 1.0);

          // Sample the 3D texture
          vec4 sample = texture(uVoxelGrid, texCoord);
          vec3 rgb = sample.rgb;
          float sigma = sample.a; // Assuming sigma is in alpha channel

          // Volume rendering equation (simplified)
          float alpha = 1.0 - exp(-sigma * stepSize * 10.0); // Scale sigma for visibility
          float weight = alpha * (1.0 - accumulatedAlpha);

          accumulatedColor += rgb * weight;
          accumulatedAlpha += weight;

          currentPosition += step;
      }

      gl_FragColor = vec4(accumulatedColor, accumulatedAlpha);
  }
`;

// Helper component to render the NeRF output texture on a plane
function NeRFPlane({ texture }: { texture: THREE.Texture }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera } = useThree();

  useFrame(() => {
    if (meshRef.current) {
      const cameraDirection = new THREE.Vector3();
      camera.getWorldDirection(cameraDirection);
      const planePosition = new THREE.Vector3().copy(camera.position).add(cameraDirection.multiplyScalar(1.5));
      meshRef.current.position.copy(planePosition);
      meshRef.current.lookAt(camera.position);
    }
  });

  return (
    <mesh ref={meshRef}>
      <planeGeometry args={[2, 2]} />
      <meshBasicMaterial map={texture} side={THREE.DoubleSide} transparent />
    </mesh>
  );
}

// Helper component to visualize camera frustums
function CameraFrustum({ poseMatrix, frustumColor = 0xff0000 }: { poseMatrix: Float32Array; frustumColor?: THREE.ColorRepresentation }) {
  const groupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    if (groupRef.current) {
      const m = mat4.clone(poseMatrix);
      const threeMatrix = new THREE.Matrix4().fromArray(Array.from(m));
      groupRef.current.applyMatrix4(threeMatrix);
      groupRef.current.updateMatrixWorld();
    }
  }, [poseMatrix]);

  const frustumGeometry = new THREE.BufferGeometry();
  const vertices = new Float32Array([
    0, 0, 0,
    -0.1, -0.1, -0.2,
    0.1, -0.1, -0.2,
    0.1, 0.1, -0.2,
    -0.1, 0.1, -0.2
  ]);
  frustumGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

  const indices = new Uint16Array([
    0, 1, 0, 2, 0, 3, 0, 4,
    1, 2, 2, 3, 3, 4, 4, 1
  ]);
  frustumGeometry.setIndex(new THREE.BufferAttribute(indices, 1));

  return (
    <group ref={groupRef}>
      <lineSegments geometry={frustumGeometry}>
        <lineBasicMaterial color={frustumColor} />
      </lineSegments>
    </group>
  );
}

// New component to visualize the voxel grid using a custom shader
function VoxelGridVisualizer({
  voxelGridTexture,
  gridMin,
  gridMax,
  maxSteps,
}: {
  voxelGridTexture: THREE.DataTexture3D;
  gridMin: THREE.Vector3;
  gridMax: THREE.Vector3;
  maxSteps: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Set up uniforms for the shader
  const uniforms = useRef({
    uVoxelGrid: { value: voxelGridTexture },
    uGridMin: { value: gridMin },
    uGridMax: { value: gridMax },
    uMaxSteps: { value: maxSteps },
  });

  // Update texture uniform if it changes
  useEffect(() => {
    if (uniforms.current.uVoxelGrid.value !== voxelGridTexture) {
      uniforms.current.uVoxelGrid.value = voxelGridTexture;
    }
    // Update maxSteps uniform if it changes
    if (uniforms.current.uMaxSteps.value !== maxSteps) {
      uniforms.current.uMaxSteps.value = maxSteps;
    }
  }, [voxelGridTexture, maxSteps]);

  return (
    <mesh ref={meshRef}>
      <boxGeometry args={[
        gridMax.x - gridMin.x,
        gridMax.y - gridMin.y,
        gridMax.z - gridMin.z,
      ]} />
      <shaderMaterial
        attach="material"
        args={[{ uniforms: uniforms.current, vertexShader: voxelVertexShader, fragmentShader: voxelFragmentShader, transparent: true }]} // transparent: true is important for volume rendering
        side={THREE.BackSide} // Render back faces first for correct ray entry point
      />
    </mesh>
  );
}

const calculateSceneBounds = (poses: CameraPoseRecord[]): {
  min: THREE.Vector3;
  max: THREE.Vector3;
} => {
  if (poses.length === 0) {
    return {
      min: new THREE.Vector3(-2, -2, -2),
      max: new THREE.Vector3(2, 2, 2)
    };
  }
  
  const positions = poses.map(pose => {
    const matrix = new THREE.Matrix4().fromArray(pose.poseMatrix);
    const position = new THREE.Vector3();
    matrix.decompose(position, new THREE.Quaternion(), new THREE.Vector3());
    return position;
  });
  
  const min = new THREE.Vector3(
    Math.min(...positions.map(p => p.x)),
    Math.min(...positions.map(p => p.y)),
    Math.min(...positions.map(p => p.z))
  );
  
  const max = new THREE.Vector3(
    Math.max(...positions.map(p => p.x)),
    Math.max(...positions.map(p => p.y)),
    Math.max(...positions.map(p => p.z))
  );
  
  // Expand bounds by 50% to include the scene
  const center = min.clone().add(max).multiplyScalar(0.5);
  const size = max.clone().sub(min);
  const expandedSize = size.multiplyScalar(1.5);
  
  return {
    min: center.clone().sub(expandedSize.clone().multiplyScalar(0.5)),
    max: center.clone().add(expandedSize.clone().multiplyScalar(0.5))
  };
};

export const NeRFRenderer: React.FC<NeRFRendererProps> = ({
  nerfModelWorkerProxy,
  cameraPoses,
  width,
  height,
  focalX = 500,
  focalY = 500,
  centerX = width / 2,
  centerY = height / 2,
  generateVoxelGridTrigger,
  useVoxelGridRendering,
  voxelGridMaxSteps,
}) => {
  const { camera } = useThree();
  const [nerfTexture] = useState(() => new THREE.DataTexture(
    new Uint8Array(HIGH_RES_WIDTH * HIGH_RES_HEIGHT * 4),
    HIGH_RES_WIDTH,
    HIGH_RES_HEIGHT,
    THREE.RGBAFormat
  ));
  nerfTexture.needsUpdate = true;

  const [voxelGridTexture, setVoxelGridTexture] = useState<THREE.DataTexture3D | null>(null);
  const [voxelGridBounds, setVoxelGridBounds] = useState<{ min: THREE.Vector3, max: THREE.Vector3 } | null>(null);

  const lastCameraPosition = useRef(new THREE.Vector3());
  const lastCameraQuaternion = useRef(new THREE.Quaternion());
  const renderTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const scaleImageData = useCallback((lowResData: Uint8Array, lowResW: number, lowResH: number, highResW: number, highResH: number): Uint8Array => {
    const highResData = new Uint8Array(highResW * highResH * 4);
    const x_ratio = lowResW / highResW;
    const y_ratio = lowResH / highResH;

    for (let y = 0; y < highResH; y++) {
      for (let x = 0; x < highResW; x++) {
        const px = Math.floor(x * x_ratio);
        const py = Math.floor(y * y_ratio);

        const lowResIndex = (py * lowResW + px) * 4;
        const highResIndex = (y * highResW + x) * 4;

        highResData[highResIndex + 0] = lowResData[lowResIndex + 0];
        highResData[highResIndex + 1] = lowResData[lowResIndex + 1];
        highResData[highResIndex + 2] = lowResData[lowResIndex + 2];
        highResData[highResIndex + 3] = lowResData[lowResIndex + 3];
      }
    }
    return highResData;
  }, []);

  // Function to perform CPU-based ray marching and update 2D texture
  const renderNeRFFrame = useCallback(async (renderWidth: number, renderHeight: number) => {
    if (!nerfModelWorkerProxy) {
      console.warn('NeRFModelWorker proxy not available.');
      return;
    }

    try {
      const c2w = mat4.create();
      camera.updateMatrixWorld();
      mat4.copy(c2w, camera.matrixWorld.elements as Float32Array);

      // Call the centralized rendering pipeline in the worker
      const { imageData, width: renderedWidth, height: renderedHeight } = await nerfModelWorkerProxy.renderNeRFImage(
        renderHeight, renderWidth,
        focalX, focalY,
        centerX, centerY,
        c2w // Pass camera matrix
      );

      const pixelArray = new Uint8Array(imageData);

      if (renderedWidth < HIGH_RES_WIDTH || renderedHeight < HIGH_RES_HEIGHT) {
        nerfTexture.image.data = scaleImageData(pixelArray, renderedWidth, renderedHeight, HIGH_RES_WIDTH, HIGH_RES_HEIGHT);
      } else {
        nerfTexture.image.data = pixelArray;
      }
      nerfTexture.needsUpdate = true;

    } catch (error) {
      console.error('Error rendering NeRF frame:', error);
    }
  }, [nerfModelWorkerProxy, camera, focalX, focalY, centerX, centerY, nerfTexture, scaleImageData]);

  // Effect to trigger voxel grid generation
  useEffect(() => {
    const generateGrid = async () => {
      if (!nerfModelWorkerProxy) return;
      console.log('Triggering voxel grid generation...');
      const VOXEL_RESOLUTION = 64; // Example resolution
      
      const bounds = calculateSceneBounds(cameraPoses);

      try {
        const { voxelData, resolution, gridMin, gridMax } = await nerfModelWorkerProxy.generateVoxelGrid(
          bounds.min.toArray(),
          bounds.max.toArray(),
          VOXEL_RESOLUTION
        );

        // Create THREE.DataTexture3D
        const texture = new THREE.DataTexture3D(
          new Float32Array(voxelData), // Use Float32Array for high precision
          resolution,
          resolution,
          resolution
        );
        texture.format = THREE.RGBAFormat; // Assuming RGBA (RGB + Sigma)
        texture.type = THREE.FloatType; // Data is Float32Array
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.needsUpdate = true;

        setVoxelGridTexture(texture);
        setVoxelGridBounds({ min: new THREE.Vector3(...gridMin), max: new THREE.Vector3(...gridMax) });
        console.log('Voxel grid texture created and ready.');
      } catch (error) {
        console.error('Failed to generate voxel grid:', error);
      }
    };

    if (generateVoxelGridTrigger > 0) {
      generateGrid();
    }
  }, [generateVoxelGridTrigger, nerfModelWorkerProxy, cameraPoses]);

  useFrame(() => {
    // Only perform CPU-based ray marching if not using voxel grid rendering
    if (!useVoxelGridRendering) {
      const currentPosition = camera.position;
      const currentQuaternion = camera.quaternion;

      const positionChanged = !currentPosition.equals(lastCameraPosition.current);
      const rotationChanged = !currentQuaternion.equals(lastCameraQuaternion.current);

      if (positionChanged || rotationChanged) {
        lastCameraPosition.current.copy(currentPosition);
        lastCameraQuaternion.current.copy(currentQuaternion);

        if (renderTimeoutRef.current) {
          clearTimeout(renderTimeoutRef.current);
        }

        renderNeRFFrame(LOW_RES_WIDTH, LOW_RES_HEIGHT);

        renderTimeoutRef.current = setTimeout(() => {
          renderNeRFFrame(HIGH_RES_WIDTH, HIGH_RES_HEIGHT);
        }, 100);
      }
    }
  });

  // Initial render for CPU-based ray marching, or if switching back from voxel grid
  useEffect(() => {
    if (!useVoxelGridRendering) {
      renderNeRFFrame(HIGH_RES_WIDTH, HIGH_RES_HEIGHT);
    }
    return () => {
      if (renderTimeoutRef.current) {
        clearTimeout(renderTimeoutRef.current);
      }
    };
  }, [renderNeRFFrame, useVoxelGridRendering]);

  return (
    <>
      {/* Conditionally render NeRFPlane or VoxelGridVisualizer */}
      {useVoxelGridRendering && voxelGridTexture && voxelGridBounds ? (
        <VoxelGridVisualizer
          voxelGridTexture={voxelGridTexture}
          gridMin={voxelGridBounds.min}
          gridMax={voxelGridBounds.max}
          maxSteps={voxelGridMaxSteps}
        />
      ) : (
        <NeRFPlane texture={nerfTexture} />
      )}

      {/* Visualize camera poses */}
      {cameraPoses.map((pose) => (
        <CameraFrustum key={pose.id} poseMatrix={pose.poseMatrix} frustumColor={0x00ff00} />
      ))}
    </>
  );
};

export const NeRFCanvas: React.FC<NeRFRendererProps> = (props) => {
  return (
    <Canvas camera={{ position: [0, 0, 3], fov: 75 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls makeDefault />
      <NeRFRenderer {...props} />
    </Canvas>
  );
};
