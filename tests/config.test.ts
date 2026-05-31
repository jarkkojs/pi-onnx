import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mergeConfig } from "../src/config.js";

describe("mergeConfig", () => {
	it("normalizes model ids and preserves explicit empty model lists", () => {
		const config = mergeConfig({
			models: [{ id: "Qwen3-0.6B-ONNX", maxTokens: 128 }],
			discovery: { enabled: false, pipelineTags: [] },
		});

		assert.equal(config.models[0]?.id, "onnx-community/Qwen3-0.6B-ONNX");
		assert.equal(config.models[0]?.maxTokens, 128);
		assert.deepEqual(config.discovery.pipelineTags, []);
	});

	it("rejects invalid enum and shape values with config paths", () => {
		assert.throws(
			() => mergeConfig({ device: "cuda" }),
			/pi-onnx: invalid config device: expected cpu \| webgpu \| wasm \| gpu/,
		);
		assert.throws(
			() => mergeConfig({ models: [null] }),
			/pi-onnx: invalid config models\[0\]: expected object/,
		);
		assert.throws(
			() => mergeConfig({ discovery: { pipelineTags: ["bad-tag"] } }),
			/pi-onnx: invalid config discovery\.pipelineTags\[0\]/,
		);
		assert.throws(
			() => mergeConfig({ tools: { embed: { pooling: "none" } } }),
			/pi-onnx: invalid config tools\.embed\.pooling/,
		);
	});
});
