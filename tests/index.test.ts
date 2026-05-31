import { it } from "node:test";
import assert from "node:assert/strict";
import { mergeModels } from "../src/index.js";

it("uses discovered dtype for pinned models without explicit dtype", () => {
	const merged = mergeModels(
		[
			{
				id: "onnx-community/gpt-oss-20b-ONNX",
				contextWindow: 131072,
			},
		],
		[
			{
				id: "onnx-community/gpt-oss-20b-ONNX",
				name: "gpt-oss-20b-ONNX",
				dtype: "q4f16",
			},
		],
	);

	assert.deepEqual(merged, [
		{
			id: "onnx-community/gpt-oss-20b-ONNX",
			name: "gpt-oss-20b-ONNX",
			contextWindow: 131072,
			dtype: "q4f16",
		},
	]);
});

it("keeps explicit pinned dtype over discovered dtype", () => {
	const merged = mergeModels(
		[
			{
				id: "onnx-community/Qwen3-0.6B-ONNX",
				name: "Pinned Qwen",
				dtype: "q8",
			},
		],
		[
			{
				id: "onnx-community/Qwen3-0.6B-ONNX",
				name: "Qwen3-0.6B-ONNX",
				dtype: "q4",
			},
		],
	);

	assert.deepEqual(merged, [
		{
			id: "onnx-community/Qwen3-0.6B-ONNX",
			name: "Pinned Qwen",
			dtype: "q8",
		},
	]);
});
