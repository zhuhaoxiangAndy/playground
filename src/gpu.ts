/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";

declare const GPUBufferUsage: any;
declare const GPUMapMode: any;

/**
 * WebGPU backend. All network math (forward pass, back propagation and the
 * weight update) runs in GPU compute shaders. Weights only travel back to
 * JavaScript once per training epoch, so the GPU stays busy.
 */

export const GPU_LINEAR = 0;
export const GPU_RELU = 1;
export const GPU_TANH = 2;
export const GPU_SIGMOID = 3;

/** Maps a JS activation function to the shader's activation id. -1 = CPU. */
export function activationId(activation: nn.ActivationFunction): number {
  if (activation === nn.Activations.LINEAR) {
    return GPU_LINEAR;
  }
  if (activation === nn.Activations.RELU) {
    return GPU_RELU;
  }
  if (activation === nn.Activations.TANH) {
    return GPU_TANH;
  }
  if (activation === nn.Activations.SIGMOID) {
    return GPU_SIGMOID;
  }
  return -1;
}

/** Maps a JS regularization function to the shader's regularization id. */
export function regularizationId(reg: nn.RegularizationFunction): number {
  if (reg === nn.RegularizationFunction.L1) {
    return 1;
  }
  if (reg === nn.RegularizationFunction.L2) {
    return 2;
  }
  return 0;
}

const FORWARD_SHADER = `
struct Params {
  n_in: u32,
  n_out: u32,
  batch: u32,
  act: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_prev: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(4) var<storage, read_write> z: array<f32>;
@group(0) @binding(5) var<storage, read_write> a: array<f32>;

fn activate(x: f32, t: u32) -> f32 {
  if (t == 1u) { return max(x, 0.0); }
  if (t == 2u) { return tanh(x); }
  if (t == 3u) { return 1.0 / (1.0 + exp(-x)); }
  return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.batch * params.n_out) { return; }
  let s = idx / params.n_out;
  let j = idx - s * params.n_out;
  var sum = biases[j];
  for (var k: u32 = 0u; k < params.n_in; k = k + 1u) {
    sum = sum + a_prev[s * params.n_in + k] * weights[j * params.n_in + k];
  }
  z[idx] = sum;
  a[idx] = activate(sum, params.act);
}
`;

const BACK_OUT_SHADER = `
struct Params { batch: u32, act: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_out: array<f32>;
@group(0) @binding(2) var<storage, read> z_out: array<f32>;
@group(0) @binding(3) var<storage, read> tgt: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_out: array<f32>;

fn activateDer(x: f32, t: u32) -> f32 {
  if (t == 1u) { return select(0.0, 1.0, x > 0.0); }
  if (t == 2u) { let o = tanh(x); return 1.0 - o * o; }
  if (t == 3u) { let o = 1.0 / (1.0 + exp(-x)); return o * (1.0 - o); }
  return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.batch) { return; }
  d_out[idx] = (a_out[idx] - tgt[idx]) * activateDer(z_out[idx], params.act);
}
`;

const BACK_HIDDEN_SHADER = `
struct Params { n_cur: u32, n_next: u32, batch: u32, act: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> d_next: array<f32>;
@group(0) @binding(2) var<storage, read> w_next: array<f32>;
@group(0) @binding(3) var<storage, read> z_cur: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_cur: array<f32>;

fn activateDer(x: f32, t: u32) -> f32 {
  if (t == 1u) { return select(0.0, 1.0, x > 0.0); }
  if (t == 2u) { let o = tanh(x); return 1.0 - o * o; }
  if (t == 3u) { let o = 1.0 / (1.0 + exp(-x)); return o * (1.0 - o); }
  return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.batch * params.n_cur) { return; }
  let s = idx / params.n_cur;
  let j = idx - s * params.n_cur;
  var sum = 0.0;
  for (var m: u32 = 0u; m < params.n_next; m = m + 1u) {
    sum = sum + d_next[s * params.n_next + m] * w_next[m * params.n_cur + j];
  }
  d_cur[idx] = sum * activateDer(z_cur[idx], params.act);
}
`;

const GRAD_SHADER = `
struct Params { n_in: u32, n_out: u32, batch: u32, pad: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_prev: array<f32>;
@group(0) @binding(2) var<storage, read> d_cur: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_w: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_b: array<f32>;
@group(0) @binding(5) var<storage, read> dead: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  let k = gid.y;
  if (j >= params.n_out || k >= params.n_in) { return; }
  let wi = j * params.n_in + k;
  if (dead[wi] != 0.0) { return; }
  var gw = 0.0;
  for (var s: u32 = 0u; s < params.batch; s = s + 1u) {
    gw = gw + d_cur[s * params.n_out + j] * a_prev[s * params.n_in + k];
  }
  grad_w[wi] = grad_w[wi] + gw;
  if (k == 0u) {
    var gb = 0.0;
    for (var s2: u32 = 0u; s2 < params.batch; s2 = s2 + 1u) {
      gb = gb + d_cur[s2 * params.n_out + j];
    }
    grad_b[j] = grad_b[j] + gb;
  }
}
`;

const UPDATE_W_SHADER = `
struct Params {
  batch: u32,
  lr: f32,
  reg_rate: f32,
  reg_type: u32,
  n: u32,
  n_in: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> grad_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> dead: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  if (dead[idx] != 0.0) { weights[idx] = 0.0; return; }
  let w = weights[idx];
  let g = grad_w[idx] / f32(params.batch);
  let data_w = w - params.lr * g;
  if (params.reg_type == 1u) {
    let sign = select(-1.0, 1.0, w >= 0.0);
    let after = data_w - params.lr * params.reg_rate * sign;
    if (data_w * after < 0.0) {
      weights[idx] = 0.0;
      dead[idx] = 1.0;
    } else {
      weights[idx] = after;
    }
  } else if (params.reg_type == 2u) {
    weights[idx] = data_w - params.lr * params.reg_rate * w;
  } else {
    weights[idx] = data_w;
  }
}
`;

const UPDATE_B_SHADER = `
struct Params { batch: u32, lr: f32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> biases: array<f32>;
@group(0) @binding(2) var<storage, read> grad_b: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&biases)) { return; }
  biases[idx] = biases[idx] - params.lr * grad_b[idx] / f32(params.batch);
}
`;

/** Maximum number of rows (samples or grid points) per dispatch buffer. */
const MAX_ROWS = 10000;

export class GpuTrainer {
  private device: any;
  private layers: number[] = [];
  private activation = GPU_TANH;
  private outputActivation = GPU_TANH;
  private regType = 0;

  private forwardPipeline: any;
  private backOutPipeline: any;
  private backHiddenPipeline: any;
  private gradPipeline: any;
  private updateWPipeline: any;
  private updateBPipeline: any;

  private weights: any[] = [];
  private biases: any[] = [];
  private gradWeights: any[] = [];
  private gradBiases: any[] = [];
  private dead: any[] = [];
  private activations: any[] = [];
  private zs: any[] = [];
  private deltas: any[] = [];
  private target: any;

  private forwardUniforms: any[] = [];
  private backOutUniform: any;
  private backHiddenUniforms: any[] = [];
  private gradUniforms: any[] = [];
  private updateWUniforms: any[] = [];
  private updateBUniforms: any[] = [];

  private forwardBindGroups: any[] = [];
  private backOutBindGroup: any;
  private backHiddenBindGroups: any[] = [];
  private gradBindGroups: any[] = [];
  private updateWBindGroups: any[] = [];
  private updateBBindGroups: any[] = [];

  private zeroGradWeights: any[] = [];
  private zeroGradBiases: any[] = [];

  /** Returns a trainer, or null when WebGPU is unavailable. */
  static async create(): Promise<GpuTrainer> {
    let gpu = (navigator as any).gpu;
    if (gpu == null) {
      return null;
    }
    try {
      let adapter = await gpu.requestAdapter();
      if (adapter == null) {
        return null;
      }
      let device = await adapter.requestDevice();
      let trainer = new GpuTrainer(device);
      let info = (adapter as any).info;
      if (info != null && info.vendor) {
        trainer.adapterName = info.vendor +
            (info.architecture ? "/" + info.architecture : "");
      }
      return trainer;
    } catch (e) {
      return null;
    }
  }

  /** Human readable name of the adapter, for the status line. */
  private adapterName = "WebGPU";
  getAdapterName(): string {
    return this.adapterName;
  }

  private constructor(device: any) {
    this.device = device;
    this.forwardPipeline = this.createPipeline(FORWARD_SHADER);
    this.backOutPipeline = this.createPipeline(BACK_OUT_SHADER);
    this.backHiddenPipeline = this.createPipeline(BACK_HIDDEN_SHADER);
    this.gradPipeline = this.createPipeline(GRAD_SHADER);
    this.updateWPipeline = this.createPipeline(UPDATE_W_SHADER);
    this.updateBPipeline = this.createPipeline(UPDATE_B_SHADER);
  }

  private createPipeline(code: string): any {
    let module = this.device.createShaderModule({code: code});
    return this.device.createComputePipeline({
      layout: "auto",
      compute: {module: module, entryPoint: "main"}
    });
  }

  private createBuffer(bytes: number, usage: number): any {
    return this.device.createBuffer({size: Math.max(4, bytes), usage: usage});
  }

  private storageBytes(elements: number): number {
    return Math.max(4, elements * 4);
  }

  /** Allocates buffers for the given network shape. */
  setNetwork(network: nn.Node[][], activation: number,
      outputActivation: number, regType: number) {
    let shape: number[] = network.map(layer => layer.length);
    this.activation = activation;
    this.outputActivation = outputActivation;
    this.regType = regType;

    let shapeChanged = shape.length !== this.layers.length;
    for (let i = 0; !shapeChanged && i < shape.length; i++) {
      shapeChanged = shape[i] !== this.layers[i];
    }
    if (shapeChanged) {
      this.destroyBuffers();
      this.layers = shape;
      this.allocateBuffers();
    }
    this.uploadWeights(network);
  }

  private destroyBuffers() {
    let all = ([] as any[]).concat(
        this.weights, this.biases, this.gradWeights, this.gradBiases,
        this.dead, this.activations, this.zs, this.deltas,
        this.forwardUniforms, this.backHiddenUniforms, this.gradUniforms,
        this.updateWUniforms, this.updateBUniforms,
        [this.target, this.backOutUniform]);
    all.forEach(buffer => {
      if (buffer != null) {
        buffer.destroy();
      }
    });
    this.weights = [];
    this.biases = [];
    this.gradWeights = [];
    this.gradBiases = [];
    this.dead = [];
    this.activations = [];
    this.zs = [];
    this.deltas = [];
    this.forwardUniforms = [];
    this.backOutUniform = null;
    this.backHiddenUniforms = [];
    this.gradUniforms = [];
    this.updateWUniforms = [];
    this.updateBUniforms = [];
    this.forwardBindGroups = [];
    this.backOutBindGroup = null;
    this.backHiddenBindGroups = [];
    this.gradBindGroups = [];
    this.updateWBindGroups = [];
    this.updateBBindGroups = [];
    this.zeroGradWeights = [];
    this.zeroGradBiases = [];
  }

  private allocateBuffers() {
    let S = GPUBufferUsage.STORAGE;
    let C = GPUBufferUsage.COPY_DST;
    let numLayers = this.layers.length;

    for (let l = 0; l < numLayers; l++) {
      let n = this.layers[l];
      this.activations[l] = this.createBuffer(
          this.storageBytes(MAX_ROWS * n), S | C | GPUBufferUsage.COPY_SRC);
      if (l >= 1) {
        let nIn = this.layers[l - 1];
        let weightCount = n * nIn;
        this.weights[l] = this.createBuffer(this.storageBytes(weightCount), S | C | GPUBufferUsage.COPY_SRC);
        this.biases[l] = this.createBuffer(this.storageBytes(n), S | C | GPUBufferUsage.COPY_SRC);
        this.gradWeights[l] = this.createBuffer(this.storageBytes(weightCount), S | C);
        this.gradBiases[l] = this.createBuffer(this.storageBytes(n), S | C);
        this.dead[l] = this.createBuffer(this.storageBytes(weightCount), S | C | GPUBufferUsage.COPY_SRC);
        this.zs[l] = this.createBuffer(this.storageBytes(MAX_ROWS * n), S | C);
        this.deltas[l] = this.createBuffer(this.storageBytes(MAX_ROWS * n), S | C);
      }
    }
    this.target = this.createBuffer(this.storageBytes(MAX_ROWS), S | C);

    this.zeroGradWeights = [];
    this.zeroGradBiases = [];
    for (let l = 1; l < numLayers; l++) {
      let weightCount = this.layers[l] * this.layers[l - 1];
      this.zeroGradWeights[l] = new Float32Array(weightCount);
      this.zeroGradBiases[l] = new Float32Array(this.layers[l]);
      this.forwardUniforms[l] = this.createBuffer(16, GPUBufferUsage.UNIFORM | C);
      this.gradUniforms[l] = this.createBuffer(16, GPUBufferUsage.UNIFORM | C);
      this.updateWUniforms[l] = this.createBuffer(32, GPUBufferUsage.UNIFORM | C);
      this.updateBUniforms[l] = this.createBuffer(8, GPUBufferUsage.UNIFORM | C);
    }
    this.backOutUniform = this.createBuffer(8, GPUBufferUsage.UNIFORM | C);
    for (let l = 1; l < numLayers - 1; l++) {
      this.backHiddenUniforms[l] = this.createBuffer(16, GPUBufferUsage.UNIFORM | C);
    }

    // Build the bind groups once, they only reference buffers.
    for (let l = 1; l < numLayers; l++) {
      this.forwardBindGroups[l] = this.device.createBindGroup({
        layout: this.forwardPipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: {buffer: this.forwardUniforms[l]}},
          {binding: 1, resource: {buffer: this.activations[l - 1]}},
          {binding: 2, resource: {buffer: this.weights[l]}},
          {binding: 3, resource: {buffer: this.biases[l]}},
          {binding: 4, resource: {buffer: this.zs[l]}},
          {binding: 5, resource: {buffer: this.activations[l]}}
        ]
      });
      this.gradBindGroups[l] = this.device.createBindGroup({
        layout: this.gradPipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: {buffer: this.gradUniforms[l]}},
          {binding: 1, resource: {buffer: this.activations[l - 1]}},
          {binding: 2, resource: {buffer: this.deltas[l]}},
          {binding: 3, resource: {buffer: this.gradWeights[l]}},
          {binding: 4, resource: {buffer: this.gradBiases[l]}},
          {binding: 5, resource: {buffer: this.dead[l]}}
        ]
      });
      this.updateWBindGroups[l] = this.device.createBindGroup({
        layout: this.updateWPipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: {buffer: this.updateWUniforms[l]}},
          {binding: 1, resource: {buffer: this.weights[l]}},
          {binding: 2, resource: {buffer: this.gradWeights[l]}},
          {binding: 3, resource: {buffer: this.dead[l]}}
        ]
      });
      this.updateBBindGroups[l] = this.device.createBindGroup({
        layout: this.updateBPipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: {buffer: this.updateBUniforms[l]}},
          {binding: 1, resource: {buffer: this.biases[l]}},
          {binding: 2, resource: {buffer: this.gradBiases[l]}}
        ]
      });
    }
    let outLayer = numLayers - 1;
    this.backOutBindGroup = this.device.createBindGroup({
      layout: this.backOutPipeline.getBindGroupLayout(0),
      entries: [
        {binding: 0, resource: {buffer: this.backOutUniform}},
        {binding: 1, resource: {buffer: this.activations[outLayer]}},
        {binding: 2, resource: {buffer: this.zs[outLayer]}},
        {binding: 3, resource: {buffer: this.target}},
        {binding: 4, resource: {buffer: this.deltas[outLayer]}}
      ]
    });
    for (let l = 1; l < numLayers - 1; l++) {
      this.backHiddenBindGroups[l] = this.device.createBindGroup({
        layout: this.backHiddenPipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: {buffer: this.backHiddenUniforms[l]}},
          {binding: 1, resource: {buffer: this.deltas[l + 1]}},
          {binding: 2, resource: {buffer: this.weights[l + 1]}},
          {binding: 3, resource: {buffer: this.zs[l]}},
          {binding: 4, resource: {buffer: this.deltas[l]}}
        ]
      });
    }
  }

  /** Copies the JS network's weights and biases into GPU buffers. */
  uploadWeights(network: nn.Node[][]) {
    let queue = this.device.queue;
    for (let l = 1; l < network.length; l++) {
      let layer = network[l];
      let n = layer.length;
      let nIn = network[l - 1].length;
      let weights = new Float32Array(n * nIn);
      let biases = new Float32Array(n);
      let dead = new Float32Array(n * nIn);
      for (let j = 0; j < n; j++) {
        biases[j] = layer[j].bias;
        for (let k = 0; k < nIn; k++) {
          let link = layer[j].inputLinks[k];
          weights[j * nIn + k] = link.weight;
          dead[j * nIn + k] = link.isDead ? 1 : 0;
        }
      }
      queue.writeBuffer(this.weights[l], 0, weights);
      queue.writeBuffer(this.biases[l], 0, biases);
      queue.writeBuffer(this.dead[l], 0, dead);
      queue.writeBuffer(this.gradWeights[l], 0, this.zeroGradWeights[l]);
      queue.writeBuffer(this.gradBiases[l], 0, this.zeroGradBiases[l]);
    }
  }

  /** Copies the GPU weights back into the JS network. */
  async syncWeights(network: nn.Node[][]) {
    let specs: {buffer: any, elements: number}[] = [];
    for (let l = 1; l < network.length; l++) {
      specs.push({buffer: this.weights[l],
          elements: network[l].length * network[l - 1].length});
      specs.push({buffer: this.biases[l], elements: network[l].length});
      specs.push({buffer: this.dead[l],
          elements: network[l].length * network[l - 1].length});
    }
    let read = await this.readBuffers(specs);
    let index = 0;
    for (let l = 1; l < network.length; l++) {
      let layer = network[l];
      let n = layer.length;
      let nIn = network[l - 1].length;
      let weights = read[index++];
      let biases = read[index++];
      let dead = read[index++];
      for (let j = 0; j < n; j++) {
        let node = layer[j];
        node.bias = biases[j];
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
        for (let k = 0; k < nIn; k++) {
          let link = node.inputLinks[k];
          link.weight = weights[j * nIn + k];
          link.isDead = dead[j * nIn + k] > 0.5;
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }

  private async readBuffers(
      specs: {buffer: any, elements: number}[]): Promise<Float32Array[]> {
    let totalBytes = 0;
    specs.forEach(spec => {
      totalBytes += this.storageBytes(spec.elements);
    });
    let staging = this.device.createBuffer({
      size: totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    let encoder = this.device.createCommandEncoder();
    let offset = 0;
    specs.forEach(spec => {
      let bytes = this.storageBytes(spec.elements);
      encoder.copyBufferToBuffer(spec.buffer, 0, staging, offset, bytes);
      offset += bytes;
    });
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    let mapped = new Float32Array(staging.getMappedRange());
    let result: Float32Array[] = [];
    offset = 0;
    specs.forEach(spec => {
      let bytes = this.storageBytes(spec.elements);
      let start = offset / 4;
      result.push(mapped.slice(start, start + spec.elements));
      offset += bytes;
    });
    staging.unmap();
    staging.destroy();
    return result;
  }

  private writeU32(buffer: any, values: number[]) {
    this.device.queue.writeBuffer(buffer, 0, new Uint32Array(values));
  }

  /** Runs one epoch of training on the GPU. */
  async trainEpoch(inputs: Float32Array, targets: Float32Array,
      batchSize: number, learningRate: number, regularizationRate: number,
      usage: number) {
    let nIn = this.layers[0];
    let numSamples = targets.length;
    let window = Math.max(1, Math.floor(batchSize));
    let chunk = Math.max(1, Math.floor(window * usage / 100));

    for (let start = 0; start < numSamples; start += window) {
      let windowSize = Math.min(window, numSamples - start);
      for (let l = 1; l < this.layers.length; l++) {
        this.device.queue.writeBuffer(this.gradWeights[l], 0,
            this.zeroGradWeights[l]);
        this.device.queue.writeBuffer(this.gradBiases[l], 0,
            this.zeroGradBiases[l]);
      }
      for (let offset = 0; offset < windowSize; offset += chunk) {
        let rows = Math.min(chunk, windowSize - offset);
        let rowStart = start + offset;
        this.device.queue.writeBuffer(this.activations[0], 0,
            inputs.subarray(rowStart * nIn, (rowStart + rows) * nIn));
        this.device.queue.writeBuffer(this.target, 0,
            targets.subarray(rowStart, rowStart + rows));
        let encoder = this.device.createCommandEncoder();
        this.encodeForward(encoder, rows);
        this.encodeBackward(encoder, rows);
        this.encodeGradients(encoder, rows);
        this.device.queue.submit([encoder.finish()]);
      }
      let encoder = this.device.createCommandEncoder();
      this.encodeUpdate(encoder, windowSize, learningRate, regularizationRate);
      this.device.queue.submit([encoder.finish()]);
    }
    // Wait until all training work is done before reading the weights back.
    await this.device.queue.onSubmittedWorkDone();
  }

  private encodeForward(encoder: any, batch: number) {
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.forwardPipeline);
    for (let l = 1; l < this.layers.length; l++) {
      let nOut = this.layers[l];
      let act = l === this.layers.length - 1 ?
          this.outputActivation : this.activation;
      this.writeU32(this.forwardUniforms[l],
          [this.layers[l - 1], nOut, batch, act]);
      pass.setBindGroup(0, this.forwardBindGroups[l]);
      pass.dispatchWorkgroups(Math.ceil(batch * nOut / 64));
    }
    pass.end();
  }

  private encodeBackward(encoder: any, batch: number) {
    let pass = encoder.beginComputePass();
    let outLayer = this.layers.length - 1;
    pass.setPipeline(this.backOutPipeline);
    this.writeU32(this.backOutUniform, [batch, this.outputActivation]);
    pass.setBindGroup(0, this.backOutBindGroup);
    pass.dispatchWorkgroups(Math.ceil(batch / 64));

    if (outLayer > 1) {
      pass.setPipeline(this.backHiddenPipeline);
      for (let l = outLayer - 1; l >= 1; l--) {
        let nCur = this.layers[l];
        this.writeU32(this.backHiddenUniforms[l],
            [nCur, this.layers[l + 1], batch, this.activation]);
        pass.setBindGroup(0, this.backHiddenBindGroups[l]);
        pass.dispatchWorkgroups(Math.ceil(batch * nCur / 64));
      }
    }
    pass.end();
  }

  private encodeGradients(encoder: any, batch: number) {
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.gradPipeline);
    for (let l = 1; l < this.layers.length; l++) {
      let nOut = this.layers[l];
      let nIn = this.layers[l - 1];
      this.writeU32(this.gradUniforms[l], [nIn, nOut, batch, 0]);
      pass.setBindGroup(0, this.gradBindGroups[l]);
      pass.dispatchWorkgroups(Math.ceil(nIn / 8), Math.ceil(nOut / 8));
    }
    pass.end();
  }

  private encodeUpdate(encoder: any, batch: number, learningRate: number,
      regularizationRate: number) {
    let pass = encoder.beginComputePass();
    for (let l = 1; l < this.layers.length; l++) {
      let nOut = this.layers[l];
      let nIn = this.layers[l - 1];
      let count = nOut * nIn;
      let params = new ArrayBuffer(24);
      let u = new Uint32Array(params);
      let f = new Float32Array(params);
      u[0] = batch;
      f[1] = learningRate;
      f[2] = regularizationRate;
      u[3] = this.regType;
      u[4] = count;
      u[5] = nIn;
      this.device.queue.writeBuffer(this.updateWUniforms[l], 0, params);
      pass.setPipeline(this.updateWPipeline);
      pass.setBindGroup(0, this.updateWBindGroups[l]);
      pass.dispatchWorkgroups(Math.ceil(count / 64));

      let biasParams = new ArrayBuffer(8);
      let bu = new Uint32Array(biasParams);
      let bf = new Float32Array(biasParams);
      bu[0] = batch;
      bf[1] = learningRate;
      this.device.queue.writeBuffer(this.updateBUniforms[l], 0, biasParams);
      pass.setPipeline(this.updateBPipeline);
      pass.setBindGroup(0, this.updateBBindGroups[l]);
      pass.dispatchWorkgroups(Math.ceil(nOut / 64));
    }
    pass.end();
  }

  /**
   * Runs the network on the given rows. Returns the output of every layer
   * (when keepLayers is true) or only the final output.
   */
  async infer(inputs: Float32Array, usage: number, keepLayers: boolean):
      Promise<{final: Float32Array, layers?: Float32Array[]}> {
    let nIn = this.layers[0];
    let rowsTotal = inputs.length / nIn;
    let chunk = Math.max(1, Math.floor(rowsTotal * usage / 100));
    let final = new Float32Array(rowsTotal);
    let outLayers: Float32Array[] = [];
    if (keepLayers) {
      for (let l = 1; l < this.layers.length; l++) {
        outLayers[l - 1] = new Float32Array(rowsTotal * this.layers[l]);
      }
    }
    for (let start = 0; start < rowsTotal; start += chunk) {
      let rows = Math.min(chunk, rowsTotal - start);
      this.device.queue.writeBuffer(this.activations[0], 0,
          inputs.subarray(start * nIn, (start + rows) * nIn));
      let encoder = this.device.createCommandEncoder();
      this.encodeForwardOnly(encoder, rows);
      let specs: {buffer: any, elements: number}[] = [];
      if (keepLayers) {
        for (let l = 1; l < this.layers.length; l++) {
          specs.push({buffer: this.activations[l],
              elements: rows * this.layers[l]});
        }
      } else {
        let outLayer = this.layers.length - 1;
        specs.push({buffer: this.activations[outLayer], elements: rows});
      }
      let staging = this.device.createBuffer({
        size: specs.reduce((sum, s) => sum + this.storageBytes(s.elements), 0),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      let byteOffset = 0;
      specs.forEach(spec => {
        let bytes = this.storageBytes(spec.elements);
        encoder.copyBufferToBuffer(spec.buffer, 0, staging, byteOffset, bytes);
        byteOffset += bytes;
      });
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      let mapped = new Float32Array(staging.getMappedRange());
      byteOffset = 0;
      if (keepLayers) {
        for (let l = 1; l < this.layers.length; l++) {
          let count = rows * this.layers[l];
          let startIndex = byteOffset / 4;
          outLayers[l - 1].set(
              mapped.subarray(startIndex, startIndex + count), start * this.layers[l]);
          byteOffset += this.storageBytes(count);
        }
      } else {
        final.set(mapped.subarray(0, rows), start);
      }
      staging.unmap();
      staging.destroy();
    }
    return {final: final, layers: keepLayers ? outLayers : undefined};
  }

  private encodeForwardOnly(encoder: any, batch: number) {
    this.encodeForward(encoder, batch);
  }

  getShape(): number[] {
    return this.layers.slice();
  }
}
