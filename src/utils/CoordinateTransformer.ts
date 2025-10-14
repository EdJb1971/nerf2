import { mat4, vec3, quat, mat3 } from 'gl-matrix';

export class CoordinateTransformer {
  /**
   * Converts DeviceOrientationEvent angles (alpha, beta, gamma) to a 4x4 camera-to-world matrix.
   * Assumes a standard OpenGL/Three.js camera coordinate system (Y-up, Z-backward).
   * DeviceOrientationEvent angles are in degrees, Z-X-Y intrinsic rotations.
   * alpha: rotation around Z axis (yaw), beta: rotation around X axis (pitch), gamma: rotation around Y axis (roll).
   *
   * The function calculates the device's orientation relative to the Earth frame,
   * then inverts it to get the Earth-to-device (or world-to-camera) rotation.
   * Finally, it applies a fixed alignment rotation to transform the device's
   * camera coordinate system (typically Y-up, Z-forward) to the desired
   * OpenGL/Three.js camera coordinate system (Y-up, Z-backward).
   *
   * @param alpha Rotation around Z axis (yaw), in degrees.
   * @param beta Rotation around X axis (pitch), in degrees.
   * @param gamma Rotation around Y axis (roll), in degrees.
   * @returns A 4x4 camera-to-world transformation matrix (mat4).
   */
  deviceOrientationToCameraMatrix(alpha: number, beta: number, gamma: number): mat4 {
    const out = mat4.create();
    const q_device_orientation = quat.create();

    // Convert degrees to radians
    const alphaRad = alpha * Math.PI / 180;
    const betaRad = beta * Math.PI / 180;
    const gammaRad = gamma * Math.PI / 180;

    // DeviceOrientationEvent uses Z-X-Y intrinsic rotations.
    // gl-matrix quat.fromEuler(out, x, y, z, order)
    // x = beta (rotation around X)
    // y = gamma (rotation around Y)
    // z = alpha (rotation around Z)
    // Order is 'ZXY'
    quat.fromEuler(q_device_orientation, betaRad, gammaRad, alphaRad, 'zxy');

    // q_device_orientation represents the rotation from Earth frame to Device frame.
    // This is R_earth_to_device.
    // We need the camera-to-world rotation.
    // R_camera_to_world = R_device_to_world * R_camera_to_device
    // R_device_to_world is the inverse of R_earth_to_device.
    const q_device_to_world = quat.invert(quat.create(), q_device_orientation);

    // Now, define the fixed rotation from the device's camera frame to the OpenGL camera frame.
    // Device camera (Y-up, Z-forward) to OpenGL camera (Y-up, Z-backward)
    // This is a 180-degree rotation around the X-axis.
    const q_device_camera_to_opengl_camera = quat.fromEuler(quat.create(), 180, 0, 0); // Rotate 180 degrees around X

    // The final camera-to-world rotation:
    // Apply the device-to-world rotation, then transform the camera's local axes
    // from device convention to OpenGL convention.
    quat.multiply(q_device_to_world, q_device_to_world, q_device_camera_to_opengl_camera);

    // Create the 4x4 matrix. For initial pose, assume origin (0,0,0) translation.
    mat4.fromRotationTranslation(out, q_device_to_world, vec3.fromValues(0, 0, 0));

    return out;
  }

  /**
   * Converts OpenCV's 3x3 rotation matrix (R) and 3x1 translation vector (t)
   * (representing a relative transformation from camera 1 to camera 2, where
   * R and t transform points from cam1 to cam2: P2 = R*P1 + t)
   * into a 4x4 camera-to-world matrix for camera 2, assuming camera 1 is at identity
   * and the target coordinate system is OpenGL/Three.js (Y-up, Z-backward).
   *
   * @param R_ocv A 3x3 rotation matrix from OpenCV (flat array of 9 numbers).
   * @param t_ocv A 3x1 translation vector from OpenCV (flat array of 3 numbers).
   * @returns A 4x4 camera-to-world transformation matrix (mat4).
   */
  opencvRTToCameraMatrix(R_ocv: number[], t_ocv: number[]): mat4 {
    const out = mat4.create();
    const R_mat3 = mat3.fromValues(
      R_ocv[0], R_ocv[1], R_ocv[2],
      R_ocv[3], R_ocv[4], R_ocv[5],
      R_ocv[6], R_ocv[7], R_ocv[8]
    );
    const t_vec3 = vec3.fromValues(t_ocv[0], t_ocv[1], t_ocv[2]);

    // Step 1: Get the camera-to-world transformation in OpenCV's coordinate system.
    // If [R_ocv | t_ocv] is the view matrix (world-to-camera) for cam2 relative to cam1,
    // then the camera-to-world matrix is its inverse: [R_ocv.T | -R_ocv.T * t_ocv].
    const R_inv = mat3.create();
    mat3.transpose(R_inv, R_mat3); // R.T

    const t_inv = vec3.create();
    // t_inv = -R.T * t
    // First calculate R.T * t
    vec3.transformMat3(t_inv, t_vec3, R_inv);
    vec3.negate(t_inv, t_inv);

    // Step 2: Define the transformation matrix from OpenCV camera coordinates
    // (X-right, Y-down, Z-forward) to OpenGL camera coordinates (X-right, Y-up, Z-backward).
    // This is a rotation around X by 180 degrees.
    const ocvToGlRotation = mat3.fromValues(
      1, 0, 0,
      0, -1, 0,
      0, 0, -1
    );

    // Step 3: Apply this transformation to R_inv and t_inv.
    // R_gl = ocvToGlRotation * R_inv
    // t_gl = ocvToGlRotation * t_inv
    const R_gl = mat3.create();
    mat3.multiply(R_gl, ocvToGlRotation, R_inv);

    const t_gl = vec3.create();
    vec3.transformMat3(t_gl, t_inv, ocvToGlRotation); // Apply rotation to translation vector

    // Step 4: Construct the final 4x4 camera-to-world matrix.
    mat4.fromRotationTranslation(out, quat.fromMat3(quat.create(), R_gl), t_gl);

    return out;
  }
}

export const coordinateTransformer = new CoordinateTransformer();
