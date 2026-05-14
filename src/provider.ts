// SPDX-License-Identifier: MIT
// Copyright (C) Jarkko Sakkinen 2026

import type {
	Api,
	AssistantMessage,
	AssistantMessageEventStream,
	Context,
	Message,
	Model,
	SimpleStreamOptions,
	StopReason,
	ToolResultMessage,
} from "@earendil-works/pi-ai";
import { calculateCost, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { hubPath, type Config, type Dtype, type ModelEntry } from "./config.js";
import { configureRuntime, getTextStreamer, loadPipeline } from "./runtime.js";

export const ONNX_API: Api = "onnx";
export const ONNX_PROVIDER = "onnx";

interface ChatMessage {
	role: "system" | "user" | "assistant";
	content: string;
}

function flattenContent(content: Message["content"]): string {
	if (typeof content === "string") return content;
	return content
		.map((block) => {
			if (block.type === "text") return block.text;
			if (block.type === "image") return "[image omitted]";
			return "";
		})
		.filter((s) => s.length > 0)
		.join("\n");
}

function summarizeToolResult(msg: ToolResultMessage): string {
	const text = msg.content
		.map((c) => (c.type === "text" ? c.text : "[image omitted]"))
		.join("\n");
	const prefix = msg.isError ? "Tool error" : "Tool result";
	return `${prefix} for call ${msg.toolCallId}:\n${text}`;
}

function toChatMessages(systemPrompt: string | undefined, messages: Message[]): ChatMessage[] {
	const chat: ChatMessage[] = [];
	if (systemPrompt && systemPrompt.trim().length > 0) {
		chat.push({ role: "system", content: systemPrompt });
	}

	for (const msg of messages) {
		if (msg.role === "user") {
			const text = flattenContent(msg.content);
			if (text.trim().length > 0) chat.push({ role: "user", content: text });
		} else if (msg.role === "assistant") {
			const parts: string[] = [];
			for (const block of msg.content) {
				if (block.type === "text" && block.text.trim().length > 0) {
					parts.push(block.text);
				} else if (block.type === "toolCall") {
					parts.push(`[called tool ${block.name} with ${JSON.stringify(block.arguments)}]`);
				}
			}
			if (parts.length > 0) chat.push({ role: "assistant", content: parts.join("\n") });
		} else if (msg.role === "toolResult") {
			chat.push({ role: "user", content: summarizeToolResult(msg) });
		}
	}

	return chat;
}

function findModelEntry(config: Config, modelId: string): ModelEntry | undefined {
	const target = hubPath(modelId);
	return config.models.find((m) => m.id === target);
}

function resolveDtype(config: Config, entry: ModelEntry | undefined): Dtype {
	return entry?.dtype ?? config.defaultDtype;
}

function mapTemperature(reasoning: SimpleStreamOptions["reasoning"] | undefined, fallback?: number): number {
	if (typeof fallback === "number") return fallback;
	switch (reasoning) {
		case "minimal":
			return 0;
		case "low":
			return 0.2;
		case "medium":
			return 0.5;
		case "high":
		case "xhigh":
			return 0.8;
		default:
			return 0.2;
	}
}

export function createOnnxStreamFunction(config: Config) {
	return function streamOnnx(
		model: Model<Api>,
		context: Context,
		options?: SimpleStreamOptions,
	): AssistantMessageEventStream {
		const stream = createAssistantMessageEventStream();

		(async () => {
			const output: AssistantMessage = {
				role: "assistant",
				content: [],
				api: model.api,
				provider: model.provider,
				model: model.id,
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			let contentIndex = -1;
			let accumulated = "";

			stream.push({ type: "start", partial: output });
			try {
				await configureRuntime(config);

				const entry = findModelEntry(config, model.id);
				const dtype = resolveDtype(config, entry);
				const fullModelId = hubPath(model.id);

				const { pipeline } = await loadPipeline("text-generation", fullModelId, {
					device: config.device,
					dtype,
				});

				const tokenizer = pipeline.tokenizer;
				const chat = toChatMessages(context.systemPrompt, context.messages);

				let inputTokens = 0;
				try {
					const encoded = tokenizer.apply_chat_template(chat as any, {
						add_generation_prompt: true,
						tokenize: true,
					});
					if (encoded && typeof (encoded as { length?: number }).length === "number") {
						inputTokens = (encoded as { length: number }).length;
					}
				} catch {
					inputTokens = 0;
				}
				output.usage.input = inputTokens;

				output.content.push({ type: "text", text: "" });
				contentIndex = output.content.length - 1;
				stream.push({ type: "text_start", contentIndex, partial: output });

				const streamer = await getTextStreamer(tokenizer, (chunk: string) => {
					if (!chunk) return;
					accumulated += chunk;
					(output.content[contentIndex] as { text: string }).text = accumulated;
					stream.push({
						type: "text_delta",
						contentIndex,
						delta: chunk,
						partial: output,
					});
				});

				const maxNewTokens = options?.maxTokens ?? entry?.maxTokens ?? model.maxTokens ?? 1024;
				const temperature = mapTemperature(options?.reasoning, options?.temperature);
				const doSample = temperature > 0;

				const abortPromise = new Promise<never>((_, reject) => {
					if (!options?.signal) return;
					if (options.signal.aborted) {
						reject(new Error("Request was aborted"));
						return;
					}
					options.signal.addEventListener(
						"abort",
						() => reject(new Error("Request was aborted")),
						{ once: true },
					);
				});

				const generatePromise = pipeline(chat as any, {
					max_new_tokens: maxNewTokens,
					do_sample: doSample,
					temperature: doSample ? temperature : undefined,
					streamer,
					return_full_text: false,
				}) as Promise<unknown>;

				await (options?.signal ? Promise.race([generatePromise, abortPromise]) : generatePromise);

				let outputTokens = 0;
				try {
					const encoded = tokenizer.encode(accumulated);
					if (Array.isArray(encoded)) outputTokens = encoded.length;
					else if (encoded && typeof (encoded as { length?: number }).length === "number") {
						outputTokens = (encoded as { length: number }).length;
					}
				} catch {
					outputTokens = Math.ceil(accumulated.length / 4);
				}

				output.usage.output = outputTokens;
				output.usage.totalTokens = output.usage.input + output.usage.output;
				calculateCost(model, output.usage);

				stream.push({
					type: "text_end",
					contentIndex,
					content: accumulated,
					partial: output,
				});

				if (outputTokens >= maxNewTokens) output.stopReason = "length";

				stream.push({
					type: "done",
					reason: output.stopReason as Extract<StopReason, "stop" | "length" | "toolUse">,
					message: output,
				});
				stream.end();
			} catch (error) {
				const aborted = options?.signal?.aborted === true;
				output.stopReason = aborted ? "aborted" : "error";
				output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
				if (contentIndex >= 0) {
					stream.push({
						type: "text_end",
						contentIndex,
						content: accumulated,
						partial: output,
					});
				}
				stream.push({ type: "error", reason: output.stopReason, error: output });
				stream.end();
			}
		})();

		return stream;
	};
}
