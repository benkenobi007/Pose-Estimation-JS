
self.importScripts('tf.min.js');
self.importScripts('posenet.min.js');
self.importScripts('facemesh.min.js');


// ML models
let facemeshModel;
let poseModel;
let minPoseConfidence = 0.15;
let minPartConfidence = 0.1;
let nmsRadius = 30.0;

const defaultPoseNetArchitecture = 'MobileNetV1';
const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 200;

var loaded = false

async function initModels() {
    // setStatusText('Loading PoseNet model...');
    console.log('Loading PoseNet model...')
    poseModel = await posenet_module.load({
        architecture: defaultPoseNetArchitecture,
        outputStride: defaultStride,
        inputResolution: defaultInputResolution,
        multiplier: defaultMultiplier,
        quantBytes: defaultQuantBytes
    });
    console.log('Loading FaceMesh model...');
    facemeshModel = await facemesh_module.load();
    loaded = true;
}

initModels();

this.onmessage = async function handler(e) {
    if (!loaded) {
        return;
    }

    console.log("in worker !");

    if (e.data.input !== 'undefined') {
        console.log("Got input in worker, runing models");
        var faceDetection = await facemeshModel.estimateFaces(input, false, false);
        let all_poses = await poseModel.estimatePoses(video, {
            flipHorizontal: true,
            decodingMethod: 'multi-person',
            maxDetections: 1,
            scoreThreshold: minPartConfidence,
            nmsRadius: nmsRadius
        });

        var message = {
            faceDetection: faceDetection,
            all_poses: all_poses
        };

        this.postMessage(message, [input]);

    }

}