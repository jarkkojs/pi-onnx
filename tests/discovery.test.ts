import { it } from "node:test";
import assert from "node:assert/strict";
import { discoverOnnxCommunityModels } from "../src/discovery.js";

function response(body: unknown): Response {
	return {
		ok: true,
		json: async () => body,
	} as Response;
}

it("discovers only compatible Transformers.js text-generation models", async () => {
	const previousFetch = globalThis.fetch;
	const seenUrls: string[] = [];
	globalThis.fetch = (async (input: Parameters<typeof fetch>[0]) => {
		seenUrls.push(String(input));
		return response([
			{
				id: "onnx-community/gpt-oss-20b-ONNX",
				library_name: "transformers.js",
				pipeline_tag: "text-generation",
				siblings: [{ rfilename: "onnx/model_q4f16.onnx" }],
			},
			{
				id: "onnx-community/Qwen3.6-27B-Onnx",
				library_name: "onnxruntime-genai",
				pipeline_tag: "image-text-to-text",
				siblings: [{ rfilename: "model.onnx" }],
			},
			{
				id: "onnx-community/tokenizer-only",
				library_name: "transformers.js",
				pipeline_tag: "text-generation",
				siblings: [{ rfilename: "tokenizer.json" }],
			},
			{
				id: "someone-else/model",
				library_name: "transformers.js",
				pipeline_tag: "text-generation",
				siblings: [{ rfilename: "onnx/model_q4.onnx" }],
			},
		]);
	}) as typeof fetch;

	try {
		const discovered = await discoverOnnxCommunityModels({
			limit: 10,
			pipelineTags: ["text-generation", "image-text-to-text"],
		});

		assert.deepEqual(discovered, [
			{
				id: "onnx-community/gpt-oss-20b-ONNX",
				name: "gpt-oss-20b-ONNX",
				dtype: "q4f16",
			},
		]);
		assert.equal(new URL(seenUrls[0]!).searchParams.get("full"), "true");
	} finally {
		globalThis.fetch = previousFetch;
	}
});

it("prefers q4 when multiple supported dtypes exist", async () => {
	const previousFetch = globalThis.fetch;
	globalThis.fetch = (async () =>
		response([
			{
				modelId: "onnx-community/Qwen3-0.6B-ONNX",
				library_name: "transformers.js",
				pipeline_tag: "text-generation",
				siblings: [
					{ rfilename: "onnx/model_q4f16.onnx" },
					{ rfilename: "onnx/model_q4.onnx" },
				],
			},
		])) as typeof fetch;

	try {
		const discovered = await discoverOnnxCommunityModels({
			limit: 10,
			pipelineTags: ["text-generation"],
		});

		assert.equal(discovered[0]?.dtype, "q4");
	} finally {
		globalThis.fetch = previousFetch;
	}
});
