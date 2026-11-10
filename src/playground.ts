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
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem
} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as gpu from "./gpu";
import * as d3 from 'd3';

let mainWidth;

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;
/** Minimum horizontal distance between two hidden layers. */
const MIN_LAYER_SPACING = 68;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777", "black"]);

// GPU (WebGPU) backend. When active, the training epoch and all inference
// (loss + decision boundary) run on the GPU. Every other code path falls
// back to the CPU implementation below.
let gpuTrainer: gpu.GpuTrainer = null;
let gpuStepBusy = false;
let gpuInitStarted = false;
/** Cached input grid for the decision boundary, keyed by input count. */
let gpuGridInputs: Float32Array = null;
let gpuGridKey: string = null;
/** Bumped whenever the network is rebuilt, to cancel in-flight GPU steps. */
let gpuGeneration = 0;
let gpuResyncPending = false;
let computeStatus = "CPU";

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#data-regen-button").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset =  newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  if (!(state.defaultNeurons >= 1)) {
    state.defaultNeurons = 2;
  }
  state.defaultNeurons = Math.floor(state.defaultNeurons);

  let addHiddenLayers = (count: number, neurons: number) => {
    count = Math.max(1, Math.floor(count) || 1);
    neurons = Math.max(1, Math.floor(neurons) || 1);
    for (let i = 0; i < count; i++) {
      state.networkShape[state.numHiddenLayers] = neurons;
      state.numHiddenLayers++;
    }
    parametersChanged = true;
    reset();
  };

  /**
   * Removes the last `count` hidden layers, then takes `neurons` neurons away
   * from every remaining hidden layer (a layer always keeps at least one).
   */
  let removeHiddenLayers = (count: number, neurons: number) => {
    count = Math.max(1, Math.floor(count) || 1);
    neurons = Math.max(0, Math.floor(neurons) || 0);
    let removed = Math.min(count, state.numHiddenLayers);
    if (removed === 0 && neurons === 0) {
      return;
    }
    state.numHiddenLayers -= removed;
    state.networkShape.splice(state.numHiddenLayers);
    if (neurons > 0) {
      for (let i = 0; i < state.numHiddenLayers; i++) {
        state.networkShape[i] = Math.max(1, state.networkShape[i] - neurons);
      }
    }
    parametersChanged = true;
    reset();
  };

  d3.select("#add-layers").on("click", () => {
    addHiddenLayers(1, state.defaultNeurons);
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  d3.select("#default-neurons")
    .property("value", state.defaultNeurons)
    .on("change", function() {
      let value = Math.max(1, Math.floor(+this.value) || 1);
      state.defaultNeurons = value;
      this.value = value;
      state.serialize();
      userHasInteracted();
    });

  let batchLayers = d3.select("#batch-layers").property("value", 1);
  let batchNeurons = d3.select("#batch-neurons")
      .property("value", state.defaultNeurons);
  let addBatchLayers = () => {
    addHiddenLayers(+batchLayers.property("value"),
        +batchNeurons.property("value"));
  };
  d3.select("#batch-add-layers").on("click", addBatchLayers);
  [batchLayers, batchNeurons].forEach(input => {
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        addBatchLayers();
      }
    });
  });

  let deleteLayers = d3.select("#delete-layers").property("value", 1);
  let deleteNeurons = d3.select("#delete-neurons").property("value", 1);
  let removeBatchLayers = () => {
    removeHiddenLayers(+deleteLayers.property("value"),
        +deleteNeurons.property("value"));
  };
  d3.select("#batch-delete-layers").on("click", removeBatchLayers);
  [deleteLayers, deleteNeurons].forEach(input => {
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        removeBatchLayers();
      }
    });
  });

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function() {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let customActivationControl = d3.select("#custom-activation-control");
  let customActivationInput = d3.select("#custom-activation");
  let customActivationError = d3.select("#custom-activation-error");
  customActivationInput.property("value", state.customActivation);
  customActivationControl.style("display",
      state.activation === activations["custom"] ? null : "none");

  let applyCustomActivation = () => {
    let expression =
        (customActivationInput.property("value") as string).trim();
    let activation = nn.createCustomActivation(expression);
    if (activation == null) {
      customActivationError.style("display", "block")
          .text("Invalid expression: use JavaScript in x, e.g. x * x.");
      return;
    }
    customActivationError.style("display", "none").text("");
    state.customActivation = expression;
    activations["custom"] = activation;
    state.activation = activation;
    parametersChanged = true;
    reset();
    updateComputeStatus();
  };

  d3.select("#apply-custom-activation").on("click", applyCustomActivation);
  customActivationInput.on("keypress", () => {
    if ((d3.event as any).keyCode === 13) {
      applyCustomActivation();
    }
  });

  let activationDropdown = d3.select("#activations").on("change", function() {
    if (this.value === "custom") {
      customActivationControl.style("display", null);
      applyCustomActivation();
      return;
    }
    customActivationControl.style("display", "none");
    state.activation = activations[this.value];
    parametersChanged = true;
    reset();
    updateComputeStatus();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  let regularDropdown = d3.select("#regularizations").on("change",
      function() {
    state.regularization = regularizations[this.value];
    parametersChanged = true;
    reset();
  });
  regularDropdown.property("value",
      getKeyFromValue(regularizations, state.regularization));

  let regularRate = d3.select("#regularRate").on("change", function() {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
  });
  regularRate.property("value", state.regularizationRate);

  let problem = d3.select("#problem").on("change", function() {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    parametersChanged = true;
    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

  // Compute backend controls (CPU / GPU) and the GPU usage throttle.
  let deviceDropdown = d3.select("#compute-device").on("change", function() {
    state.computeDevice = this.value;
    state.serialize();
    userHasInteracted();
    if (state.computeDevice === "gpu") {
      initGpu();
      // Continue from the weights the JS network currently holds.
      syncGpuNetwork();
    }
    updateComputeStatus();
  });
  if (state.computeDevice !== "cpu" && state.computeDevice !== "gpu") {
    state.computeDevice = "cpu";
  }
  if (state.computeDevice === "gpu") {
    initGpu();
  }
  deviceDropdown.property("value", state.computeDevice);

  let usageSlider = d3.select("#gpu-usage").on("input", function() {
    state.gpuUsage = +this.value;
    d3.select("#gpu-usage-value").text(state.gpuUsage + "%");
    state.serialize();
    userHasInteracted();
  });
  state.gpuUsage = Math.max(10, Math.min(100, state.gpuUsage || 100));
  usageSlider.property("value", state.gpuUsage);
  d3.select("#gpu-usage-value").text(state.gpuUsage + "%");
  updateComputeStatus();

  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.

  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
          state.discretize);
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let numHiddenLayers = Math.max(0, numLayers - 2);

  // Grow the page layout when there are many hidden layers, so that the
  // plus/minus controls and the neuron canvases never overlap. The extra
  // width is distributed to the columns by the flex layout.
  let mainPart = d3.select("#main-part").node() as HTMLElement;
  mainPart.style.minWidth = "";
  let requiredWidth = featureWidth + RECT_SIZE +
      Math.max(0, numHiddenLayers - 1) * MIN_LAYER_SPACING;
  for (let i = 0; i < 5 && requiredWidth > width; i++) {
    mainPart.style.minWidth =
        `${mainPart.offsetWidth + Math.ceil(requiredWidth - width)}px`;
    width = co.offsetLeft - cf.offsetLeft;
  }
  svg.attr("width", width);

  let layerSpacing = numHiddenLayers > 1 ?
      Math.max(MIN_LAYER_SPACING,
          (width - RECT_SIZE - featureWidth) / (numHiddenLayers - 1)) : 0;
  let layerX = (layerIdx: number) => {
    if (numHiddenLayers <= 1) {
      return (featureWidth + width - RECT_SIZE) / 2;
    }
    return featureWidth + (layerIdx - 1) * layerSpacing;
  };
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);

  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerX(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerX(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2;
  node2coord[node.id] = {cx, cy};
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
  }
  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function() {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function() {
      updateHoverCard(null);
    });
  return line;
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      let input = constructInput(x, y);
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false, updatedBoundary = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  // Get the decision boundary of the network. The GPU path already filled
  // `boundary` while it was computing, so it can skip the CPU recompute.
  if (!updatedBoundary) {
    updateDecisionBoundary(network, firstStep);
  }
  let selectedId = selectedNodeId != null ?
      selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {heatmap: HeatMap, id: string}) {
    data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        state.discretize);
  });

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function oneStep(): void {
  if (useGpu()) {
    // The GPU runs a whole epoch asynchronously. Skip frames while it is
    // busy so we never overlap two epochs on the same buffers.
    if (gpuStepBusy) {
      return;
    }
    gpuStepBusy = true;
    gpuOneStep().then(() => {
      gpuStepBusy = false;
      finishGpuStep();
    }).catch((err) => {
      gpuStepBusy = false;
      console.error("GPU step failed, falling back to CPU.", err);
      computeStatus = "GPU error - using CPU";
      state.computeDevice = "cpu";
      d3.select("#compute-device").property("value", "cpu");
      updateComputeStatus();
      finishGpuStep();
      cpuOneStep();
    });
    return;
  }
  cpuOneStep();
}

/** Applies a network rebuild that had to wait for the GPU to go idle. */
function finishGpuStep() {
  if (gpuResyncPending) {
    gpuResyncPending = false;
    try {
      syncGpuNetwork();
    } catch (e) {
      console.error("Failed to resync the GPU network.", e);
    }
  }
}

function cpuOneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateUI();
}

/** True when the current network can run on the WebGPU backend. */
function useGpu(): boolean {
  return state.computeDevice === "gpu" && gpuTrainer != null && gpuCompatible();
}

/** The GPU shaders only implement the built-in activation functions. */
function gpuCompatible(): boolean {
  return network != null && state.activation != null &&
      gpu.activationId(state.activation) >= 0;
}

function getOutputActivation(): nn.ActivationFunction {
  return (state.problem === Problem.REGRESSION) ?
      nn.Activations.LINEAR : nn.Activations.TANH;
}

/** Creates the WebGPU device once, on demand. */
function initGpu() {
  if (gpuInitStarted || gpuTrainer != null) {
    return;
  }
  gpuInitStarted = true;
  computeStatus = "Initializing GPU...";
  updateComputeStatus();
  gpu.GpuTrainer.create().then((trainer) => {
    gpuTrainer = trainer;
    if (trainer == null) {
      computeStatus = "WebGPU unavailable - using CPU";
    } else {
      computeStatus = "GPU " + trainer.getAdapterName();
      if (network != null) {
        syncGpuNetwork();
      }
    }
    updateComputeStatus();
  });
}

/** Copies the current network into the GPU backend. */
function syncGpuNetwork() {
  if (gpuTrainer == null || network == null) {
    return;
  }
  let activation = gpu.activationId(state.activation);
  if (activation < 0) {
    // Custom activation: the shaders cannot evaluate it.
    return;
  }
  if (gpuStepBusy) {
    // Never reallocate GPU buffers under a step that is still in flight.
    gpuResyncPending = true;
    return;
  }
  gpuTrainer.setNetwork(network, activation,
      gpu.activationId(getOutputActivation()),
      gpu.regularizationId(state.regularization));
}

function updateComputeStatus() {
  let status = d3.select("#compute-status");
  if (status.empty()) {
    return;
  }
  let label: string;
  let active = state.computeDevice === "gpu";
  if (!active) {
    label = "CPU";
  } else if (gpuTrainer == null) {
    label = gpuInitStarted ? "WebGPU unavailable - using CPU" : "Starting GPU...";
  } else if (!gpuCompatible()) {
    label = "CPU fallback (custom activation)";
  } else {
    label = computeStatus;
  }
  status.text(label);
  d3.select("#top-controls .ui-gpu-usage").classed(
      "disabled", !active || gpuTrainer == null || !gpuCompatible());
  d3.select("#gpu-usage").property("disabled", !active);
}

/** Packs the data points into a row-major [rows, inputs] matrix. */
function buildInputMatrix(dataPoints: Example2D[],
    numInputs: number): Float32Array {
  let data = new Float32Array(dataPoints.length * numInputs);
  for (let i = 0; i < dataPoints.length; i++) {
    let input = constructInput(dataPoints[i].x, dataPoints[i].y);
    for (let k = 0; k < numInputs; k++) {
      data[i * numInputs + k] = input[k];
    }
  }
  return data;
}

/** Cached input grid for the decision boundary heat maps. */
function getGridInputs(numInputs: number): Float32Array {
  let key = constructInputIds().join(",");
  if (gpuGridInputs != null && gpuGridKey === key) {
    return gpuGridInputs;
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);
  let data = new Float32Array(DENSITY * DENSITY * numInputs);
  let row = 0;
  for (let i = 0; i < DENSITY; i++) {
    for (let j = 0; j < DENSITY; j++) {
      let input = constructInput(xScale(i), yScale(j));
      for (let k = 0; k < numInputs; k++) {
        data[row * numInputs + k] = input[k];
      }
      row++;
    }
  }
  gpuGridInputs = data;
  gpuGridKey = key;
  return data;
}

/** Mean squared error, matching nn.Errors.SQUARE. */
function squareLoss(output: Float32Array, target: Float32Array): number {
  if (target.length === 0) {
    return 0;
  }
  let loss = 0;
  for (let i = 0; i < target.length; i++) {
    let diff = output[i] - target[i];
    loss += 0.5 * diff * diff;
  }
  return loss / target.length;
}

/** Fills `boundary` from the GPU activations of the sampled grid. */
function applyGpuBoundary(layers: Float32Array[]) {
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);
  for (let i = 0; i < DENSITY; i++) {
    let x = xScale(i);
    for (let j = 0; j < DENSITY; j++) {
      let y = yScale(j);
      for (let nodeId in INPUTS) {
        if (boundary[nodeId] == null) {
          boundary[nodeId] = new Array(DENSITY);
        }
        if (boundary[nodeId][i] == null) {
          boundary[nodeId][i] = new Array(DENSITY);
        }
        boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
      }
    }
  }
  for (let l = 1; l < network.length; l++) {
    let layerOut = layers[l - 1];
    let layer = network[l];
    let n = layer.length;
    for (let row = 0; row < DENSITY * DENSITY; row++) {
      let i = Math.floor(row / DENSITY);
      let j = row - i * DENSITY;
      let base = row * n;
      for (let k = 0; k < n; k++) {
        let node = layer[k];
        if (boundary[node.id] == null) {
          boundary[node.id] = new Array(DENSITY);
        }
        if (boundary[node.id][i] == null) {
          boundary[node.id][i] = new Array(DENSITY);
        }
        boundary[node.id][i][j] = layerOut[base + k];
      }
    }
  }
}

async function gpuOneStep() {
  // Snapshot the network and a generation id so a reset mid-step can cancel
  // this work instead of writing old results into the new network.
  let gen = gpuGeneration;
  let net = network;
  iter++;
  let numInputs = constructInput(0, 0).length;
  let trainInputs = buildInputMatrix(trainData, numInputs);
  let trainTargets = new Float32Array(trainData.length);
  trainData.forEach((point, i) => {
    trainTargets[i] = point.label;
  });
  await gpuTrainer.trainEpoch(trainInputs, trainTargets, state.batchSize,
      state.learningRate, state.regularizationRate, state.gpuUsage);
  if (gen !== gpuGeneration) {
    return;
  }
  await gpuTrainer.syncWeights(net);
  if (gen !== gpuGeneration) {
    return;
  }

  // Loss for the train and test splits.
  let trainOut = await gpuTrainer.infer(trainInputs, state.gpuUsage, false);
  if (gen !== gpuGeneration) {
    return;
  }
  lossTrain = squareLoss(trainOut.final, trainTargets);
  let testInputs = buildInputMatrix(testData, numInputs);
  let testTargets = new Float32Array(testData.length);
  testData.forEach((point, i) => {
    testTargets[i] = point.label;
  });
  let testOut = await gpuTrainer.infer(testInputs, state.gpuUsage, false);
  if (gen !== gpuGeneration) {
    return;
  }
  lossTest = squareLoss(testOut.final, testTargets);

  // Decision boundaries for the heat maps.
  let grid = getGridInputs(numInputs);
  let gridOut = await gpuTrainer.infer(grid, state.gpuUsage, true);
  if (gen !== gpuGeneration) {
    return;
  }
  applyGpuBoundary(gridOut.layers);

  updateUI(false, true);
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup=false) {
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();
  gpuGeneration++;

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let outputActivation = getOutputActivation();
  network = nn.buildNetwork(shape, state.activation, outputActivation,
      state.regularization, constructInputIds(), state.initZero);
  // Keep the GPU copy in sync with the freshly (re)initialized network.
  syncGpuNetwork();
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  updateUI(true);
};

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
      .text(title.text());
      document.title = title.text();
    }
  });
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 6) / 12, h * (d.y + 6) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
          document.querySelector(`canvas[data-dataset=${dataset}]`);
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (let regDataset in regDatasets) {
      let canvas: any =
          document.querySelector(`canvas[data-regDataset=${regDataset}]`);
      let dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let numSamples = (state.problem === Problem.REGRESSION) ?
      NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
  ga('set', 'page', page);
  ga('send', 'pageview', {'sessionControl': 'start'});
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}

drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
