import { it } from "node:test";
import assert from "node:assert/strict";
import type { Api, AssistantMessageEvent, Context, Model } from "@earendil-works/pi-ai";
import { mergeConfig } from "../src/config.js";
import { createOnnxStreamFunction, type OnnxRuntimeDeps } from "../src/provider.js";

it("ends with an aborted error and ignores later streamer callbacks", async () => {
	let emitText: ((chunk: string) => void) | undefined;
	const pipeline = Object.assign(
		async (_chat: unknown, _options: unknown) => {
			emitText?.("before-abort");
			await new Promise((resolve) => setTimeout(resolve, 20));
			emitText?.("after-abort");
			return [];
		},
		{
			tokenizer: {
				apply_chat_template: () => [1, 2, 3],
				encode: (text: string) => [...text],
			},
		},
	);
	const deps: OnnxRuntimeDeps = {
		configureRuntime: async () => {},
		loadPipeline: async () => ({ pipeline }),
		getTextStreamer: async (_tokenizer, onText) => {
			emitText = onText;
			return {};
		},
	};
	const config = mergeConfig({ models: [{ id: "fake-model", maxTokens: 32 }], discovery: { enabled: false } });
	const streamOnnx = createOnnxStreamFunction(config, deps);
	const controller = new AbortController();
	const model = {
		api: "onnx",
		provider: "onnx",
		id: "fake-model",
		name: "fake-model",
		input: ["text"],
		reasoning: false,
		contextWindow: 4096,
		maxTokens: 32,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	} as Model<Api>;
	const context: Context = { messages: [{ role: "user", content: "hello", timestamp: Date.now() }] };

	const events: AssistantMessageEvent[] = [];
	for await (const event of streamOnnx(model, context, { signal: controller.signal })) {
		events.push(event);
		if (event.type === "text_delta") controller.abort();
	}

	const error = events.at(-1);
	assert.equal(error?.type, "error");
	if (error?.type === "error") {
		assert.equal(error.reason, "aborted");
	}
	assert.equal(events.filter((event) => event.type === "text_delta").length, 1);
});
