import { it } from "node:test";
import assert from "node:assert/strict";
import { mergeModels, validatePinnedModels } from "../src/index.js";

function response(body: unknown, ok = true): Response {
	return {
		ok,
		status: ok ? 200 : 503,
		statusText: ok ? "OK" : "Service Unavailable",
		json: async () => body,
	} as Response;
}

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

it("skips pinned models that are not Transformers.js text-generation repositories", async () => {
	const previousFetch = globalThis.fetch;
	globalThis.fetch = (async () =>
		response({
			id: "onnx-community/Qwen3.6-27B-Onnx",
			library_name: "onnxruntime-genai",
			pipeline_tag: "image-text-to-text",
			tags: ["onnxruntime-genai", "image-text-to-text"],
			siblings: [{ rfilename: "model.onnx" }],
		})) as typeof fetch;

	try {
		const validation = await validatePinnedModels([
			{
				id: "onnx-community/Qwen3.6-27B-Onnx",
			},
		]);

		assert.deepEqual(validation.discovered, []);
		assert.equal(validation.skipped[0]?.id, "onnx-community/Qwen3.6-27B-Onnx");
		assert.match(validation.skipped[0]?.reason ?? "", /not transformers\.js/);
	} finally {
		globalThis.fetch = previousFetch;
	}
});

it("keeps pinned models when compatibility lookup fails", async () => {
	const previousFetch = globalThis.fetch;
	globalThis.fetch = (async () => response({}, false)) as typeof fetch;

	try {
		const validation = await validatePinnedModels([
			{
				id: "onnx-community/private-or-cached-model",
			},
		]);

		assert.deepEqual(validation, { discovered: [], skipped: [] });
	} finally {
		globalThis.fetch = previousFetch;
	}
});