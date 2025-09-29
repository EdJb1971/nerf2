import * as tf from '@tensorflow/tfjs';
import { mat4 } from 'gl-matrix';

export class NeRFDataUtils {

  /**
   * Generates ray directions for each pixel in an image plane given camera intrinsics and camera-to-world matrix.
   * The camera is assumed to be at the origin looking along the -Z axis in its local coordinate system.
   *
   * @param H Image height.
   * @param W Image width.
   * @param focalX Focal length in x-direction.
   * @param focalY Focal length in y-direction.
   * @param centerX Principal point x-coordinate.
   * @param centerY Principal point y-coordinate.
   * @param c2w Camera-to-world transformation matrix (mat4).
   * @returns A tensor of ray directions in world coordinates, shape [H*W, 3].
   */
  getRayDirections(
    H: number, W: number,
    focalX: number, focalY: number,
    centerX: number, centerY: number,
    c2w: mat4
  ): tf.Tensor {
    return tf.tidy(() => {
      // Create a grid of pixel coordinates
      const i = tf.linspace(0, W - 1, W); // x-coordinates
      const j = tf.linspace(0, H - 1, H); // y-coordinates

      const [gridI, gridJ] = tf.meshgrid(i, j); // gridI: [H, W], gridJ: [H, W]

      // Normalize pixel coordinates to camera coordinates
      // (u - cx) / fx, (v - cy) / fy
      const x = gridI.sub(centerX).div(focalX); // [H, W]
      const y = gridJ.sub(centerY).div(focalY); // [H, W]

      // Stack x, y, and -1 (for z-direction in camera space)
      // The camera looks along -Z in its own coordinate system.
      // So, the direction vector in camera space is (x, y, -1).
      const ones = tf.onesLike(x).mul(-1); // [H, W]
      const directionsCam = tf.stack([x, y, ones], -1); // [H, W, 3]

      // Flatten to [H*W, 3]
      const directionsCamFlat = directionsCam.reshape([-1, 3]);

      // Transform camera directions to world directions using the rotation part of c2w
      // gl-matrix uses column-major order for mat4.
      // The 3x3 rotation matrix (R_c2w) is:
      // [c2w[0], c2w[4], c2w[8]]
      // [c2w[1], c2w[5], c2w[9]]
      // [c2w[2], c2w[6], c2w[10]]
      const R_c2w_elements = [
        c2w[0], c2w[4], c2w[8],
        c2w[1], c2w[5], c2w[9],
        c2w[2], c2w[6], c2w[10]
      ];
      const R_c2w = tf.tensor(R_c2w_elements, [3, 3]);

      // If directionsCamFlat is [N, 3] (N row vectors), and R_c2w transforms column vectors,
      // then to transform row vectors, we multiply by R_c2w.T.
      const directionsWorld = directionsCamFlat.matMul(R_c2w.transpose()); // [H*W, 3] @ [3, 3] -> [H*W, 3]

      // Normalize ray directions to unit vectors
      return directionsWorld.div(directionsWorld.norm(2, -1, true));
    });
  }

  /**
   * Generates ray origins and directions for all pixels given camera intrinsics and camera-to-world matrix.
   *
   * @param H Image height.
   * @param W Image width.
   * @param focalX Focal length in x-direction.
   * @param focalY Focal length in y-direction.
   * @param centerX Principal point x-coordinate.
   * @param centerY Principal point y-coordinate.
   * @param c2w Camera-to-world transformation matrix (mat4).
   * @returns An object containing ray origins and directions.
   *          rayOrigins: tf.Tensor of shape [H*W, 3].
   *          rayDirections: tf.Tensor of shape [H*W, 3].
   */
  getRays(
    H: number, W: number,
    focalX: number, focalY: number,
    centerX: number, centerY: number,
    c2w: mat4
  ): { rayOrigins: tf.Tensor, rayDirections: tf.Tensor } {
    return tf.tidy(() => {
      const rayDirections = this.getRayDirections(H, W, focalX, focalY, centerX, centerY, c2w);

      // Ray origins are simply the camera position (translation part of c2w)
      // gl-matrix mat4 is column-major, so translation is at indices 12, 13, 14.
      const cameraOrigin = tf.tensor1d([c2w[12], c2w[13], c2w[14]]); // [3]
      const rayOrigins = tf.tile(cameraOrigin.expandDims(0), [H * W, 1]); // [H*W, 3]

      return { rayOrigins, rayDirections };
    });
  }

  /**
   * Samples points along rays using stratified sampling.
   *
   * @param rayOrigins Tensor of ray origins, shape [N_rays, 3].
   * @param rayDirections Tensor of ray directions, shape [N_rays, 3].
   * @param near Near bound for sampling.
   * @param far Far bound for sampling.
   * @param N_samples Number of samples per ray.
   * @param perturb If true, add uniform noise for stratified sampling.
   * @returns An object containing sampled points and depth values.
   *          sampledPoints: tf.Tensor of shape [N_rays * N_samples, 3].
   *          depthValues: tf.Tensor of shape [N_rays, N_samples].
   */
  samplePointsAlongRays(
    rayOrigins: tf.Tensor,
    rayDirections: tf.Tensor,
    near: number,
    far: number,
    N_samples: number,
    perturb: boolean = true
  ): { sampledPoints: tf.Tensor, depthValues: tf.Tensor } {
    return tf.tidy(() => {
      const N_rays = rayOrigins.shape[0];

      // Generate linearly spaced depth values
      let t_vals = tf.linspace(0, 1, N_samples).mul(far - near).add(near); // [N_samples]

      if (perturb) {
        // Add uniform noise for stratified sampling
        const midpoints = t_vals.slice([0], [N_samples - 1]).add(t_vals.slice([1], [N_samples])).div(2); // [N_samples - 1]
        // Create lower and upper bounds for uniform sampling within each bin
        const lower = tf.concat([t_vals.slice([0], [1]), midpoints]); // [N_samples]
        const upper = tf.concat([midpoints, t_vals.slice([N_samples - 1], [1])]); // [N_samples]

        const t_rand = tf.randomUniform([N_samples], 0, 1); // [N_samples]
        t_vals = lower.add(upper.sub(lower).mul(t_rand)); // [N_samples]
      }

      // Expand t_vals to [N_rays, N_samples]
      const t_vals_expanded = tf.tile(t_vals.expandDims(0), [N_rays, 1]); // [N_rays, N_samples]

      // Reshape rayOrigins and rayDirections for broadcasting
      const origins_reshaped = rayOrigins.expandDims(1); // [N_rays, 1, 3]
      const directions_reshaped = rayDirections.expandDims(1); // [N_rays, 1, 3]
      const t_vals_reshaped = t_vals_expanded.expandDims(-1); // [N_rays, N_samples, 1]

      // Calculate sampled points: o + t*d
      const sampledPoints = origins_reshaped.add(directions_reshaped.mul(t_vals_reshaped)); // [N_rays, N_samples, 3]

      return {
        sampledPoints: sampledPoints.reshape([-1, 3]), // Flatten to [N_rays * N_samples, 3]
        depthValues: t_vals_expanded // [N_rays, N_samples]
      };
    });
  }

  /**
   * Performs volume rendering to convert sampled RGB and sigma values along rays into final pixel colors.
   *
   * @param rgb Tensor of predicted RGB values for sampled points, shape [N_rays, N_samples, 3].
   * @param sigma Tensor of predicted sigma (density) values for sampled points, shape [N_rays, N_samples, 1].
   * @param depthValues Tensor of depth values for sampled points, shape [N_rays, N_samples].
   * @returns An object containing rendered RGB colors, rendered depth, and weights.
   *          renderedRgb: tf.Tensor of shape [N_rays, 3].
   *          renderedDepth: tf.Tensor of shape [N_rays, 1].
   *          weights: tf.Tensor of shape [N_rays, N_samples].
   */
  volumeRender(
    rgb: tf.Tensor,
    sigma: tf.Tensor,
    depthValues: tf.Tensor
  ): { renderedRgb: tf.Tensor, renderedDepth: tf.Tensor, weights: tf.Tensor } {
    return tf.tidy(() => {
      const N_rays = rgb.shape[0];
      const N_samples = rgb.shape[1];

      // Calculate delta (distance between adjacent samples)
      // delta_i = t_{i+1} - t_i
      const delta = depthValues.slice([0, 1], [N_rays, N_samples - 1])
        .sub(depthValues.slice([0, 0], [N_rays, N_samples - 1])); // [N_rays, N_samples - 1]

      // Append a large value for the last delta (representing distance to infinity)
      const largeDelta = tf.fill([N_rays, 1], 1e10); // Or (far - depthValues[:, -1]) if 'far' is available
      const delta_full = tf.concat([delta, largeDelta], -1); // [N_rays, N_samples]

      // Calculate alpha (opacity)
      // alpha_i = 1 - exp(-sigma_i * delta_i)
      const alpha = tf.onesLike(sigma).sub(tf.exp(sigma.mul(delta_full.expandDims(-1)).neg())); // [N_rays, N_samples, 1]

      // Calculate transmittance T_i = exp(-sum(sigma_j * delta_j for j < i))
      // T_i = cumprod(1 - alpha_j + 1e-10 for j < i)
      // For numerical stability, 1e-10 is added to (1 - alpha)
      const onesMinusAlpha = tf.onesLike(alpha).sub(alpha).add(1e-10); // [N_rays, N_samples, 1]
      const T = tf.cumprod(onesMinusAlpha, 1, true); // [N_rays, N_samples, 1] (exclusive cumulative product)

      // Weights for each sample along the ray
      // weights_i = T_i * alpha_i
      const weights = T.mul(alpha); // [N_rays, N_samples, 1]

      // Rendered RGB color: sum(weights_i * rgb_i)
      const renderedRgb = weights.mul(rgb).sum(1); // [N_rays, 3]

      // Rendered depth: sum(weights_i * depth_values_i)
      const renderedDepth = weights.squeeze([-1]).mul(depthValues).sum(1).expandDims(-1); // [N_rays, 1]

      return { renderedRgb, renderedDepth, weights: weights.squeeze([-1]) }; // Squeeze weights to [N_rays, N_samples]
    });
  }
}

export const nerfDataUtils = new NeRFDataUtils();