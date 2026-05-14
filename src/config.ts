// SPDX-License-Identifier: MIT
// Copyright (C) Jarkko Sakkinen 2026

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type Dtype = "fp32" | "fp16" | "q8" | "int8" | "uint8" | "q4" | "bnb4" | "q4f16";
export type Device = "cpu" | "webgpu" | "wasm" | "gpu";

export interface ModelEntry {
	id: string;
	name?: string;
	contextWindow?: number;
	maxTokens?: number;
	dtype?: Dtype;
}

export interface EmbedToolConfig {
	enabled: boolean;
	model: string;
	pooling: "mean" | "cls" | "none";
	normalize: boolean;
}

export interface ClassifyToolConfig {
	enabled: boolean;
	model: string;
	topK: number;
}

export interface TranscribeToolConfig {
	enabled: boolean;
	model: string;
	language: string | null;
	task: "transcribe" | "translate";
}

export const HF_PREFIX = "onnx-community/";

export type PipelineTag = "text-generation" | "image-text-to-text" | "any-to-any";

export interface DiscoveryConfig {
	enabled: boolean;
	limit: number;
	pipelineTags: PipelineTag[];
}

export function stripPrefix(id: string): string {
	return id.startsWith(HF_PREFIX) ? id.slice(HF_PREFIX.length) : id;
}

export function hubPath(id: string): string {
	return id.startsWith(HF_PREFIX) ? id : HF_PREFIX + id;
}

export interface Config {
	cacheDir: string | null;
	device: Device;
	defaultDtype: Dtype;
	/** Models always shown. `[]` + discovery.enabled = discovery only. */
	models: ModelEntry[];
	discovery: DiscoveryConfig;
	tools: {
		embed: EmbedToolConfig;
		classify: ClassifyToolConfig;
		transcribe: TranscribeToolConfig;
	};
}

const DEFAULTS: Config = {
	cacheDir: null,
	device: "cpu",
	defaultDtype: "q4",
	models: [
		{
			id: "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
			name: "Qwen2.5-Coder-0.5B (ONNX, q4)",
			contextWindow: 32768,
			maxTokens: 2048,
		},
	],
	discovery: {
		enabled: true,
		limit: 50,
		pipelineTags: ["text-generation", "image-text-to-text", "any-to-any"],
	},
	tools: {
		embed: {
			enabled: true,
			model: "onnx-community/all-MiniLM-L6-v2",
			pooling: "mean",
			normalize: true,
		},
		classify: {
			enabled: false,
			model: "onnx-community/distilbert-base-uncased-finetuned-sst-2-english",
			topK: 5,
		},
		transcribe: {
			enabled: false,
			model: "onnx-community/whisper-tiny",
			language: null,
			task: "transcribe",
		},
	},
};

export function configPath(): string {
	return join(homedir(), ".pi", "agent", "pi-onnx.json");
}

export function loadConfig(): Config {
	const path = configPath();
	let raw: string;
	try {
		raw = readFileSync(path, "utf8");
	} catch {
		return DEFAULTS;
	}

	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		throw new Error(`pi-onnx: failed to parse ${path}: ${msg}`);
	}

	return mergeConfig(parsed);
}

function mergeConfig(input: unknown): Config {
	if (!input || typeof input !== "object") return DEFAULTS;
	const src = input as Record<string, unknown>;
	const tools = (src.tools as Record<string, unknown> | undefined) ?? {};

	const discovery = (src.discovery as Partial<DiscoveryConfig> | undefined) ?? {};
	const rawModels = Array.isArray(src.models) ? (src.models as ModelEntry[]) : DEFAULTS.models;
	return {
		cacheDir: typeof src.cacheDir === "string" ? src.cacheDir : DEFAULTS.cacheDir,
		device: (src.device as Device) ?? DEFAULTS.device,
		defaultDtype: (src.defaultDtype as Dtype) ?? DEFAULTS.defaultDtype,
		models: rawModels.map((m) => ({ ...m, id: hubPath(m.id) })),
		discovery: { ...DEFAULTS.discovery, ...discovery },
		tools: {
			embed: { ...DEFAULTS.tools.embed, ...(tools.embed as Partial<EmbedToolConfig> | undefined) },
			classify: { ...DEFAULTS.tools.classify, ...(tools.classify as Partial<ClassifyToolConfig> | undefined) },
			transcribe: {
				...DEFAULTS.tools.transcribe,
				...(tools.transcribe as Partial<TranscribeToolConfig> | undefined),
			},
		},
	};
}
