// SPDX-License-Identifier: MIT
// Copyright (C) Jarkko Sakkinen 2026

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { configPath, loadConfig, stripPrefix, type Config, type ModelEntry } from "./config.js";
import { discoverOnnxCommunityModels, type DiscoveredModel } from "./discovery.js";
import { createOnnxStreamFunction, ONNX_API, ONNX_PROVIDER } from "./provider.js";
import { registerAllTools } from "./tools.js";

function buildProviderConfig(
	models: ModelEntry[],
	streamSimple: ReturnType<typeof createOnnxStreamFunction>,
) {
	return {
		baseUrl: "local://onnx",
		apiKey: "onnx",
		api: ONNX_API,
		models: models.map((m) => {
			const short = stripPrefix(m.id);
			return {
				id: short,
				name: m.name ?? short,
				reasoning: false,
				input: ["text" as const],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: m.contextWindow ?? 4096,
				maxTokens: m.maxTokens ?? 1024,
			};
		}),
		streamSimple,
	};
}

function mergeModels(pinned: ModelEntry[], discovered: DiscoveredModel[]): ModelEntry[] {
	const seen = new Set(pinned.map((m) => m.id));
	const out: ModelEntry[] = [...pinned];
	for (const d of discovered) {
		if (seen.has(d.id)) continue;
		seen.add(d.id);
		out.push({ id: d.id, name: d.name });
	}
	return out;
}

export default function (pi: ExtensionAPI) {
	const config = loadConfig();
	const streamSimple = createOnnxStreamFunction(config);
	let discovered: DiscoveredModel[] = [];
	let discoveryError: string | null = null;

	pi.registerProvider(ONNX_PROVIDER, buildProviderConfig(config.models, streamSimple));

	if (config.discovery.enabled) {
		discoverOnnxCommunityModels({
			limit: config.discovery.limit,
			pipelineTags: config.discovery.pipelineTags,
		})
			.then((found) => {
				discovered = found;
				const merged = mergeModels(config.models, found);
				if (merged.length === config.models.length) return;
				pi.registerProvider(ONNX_PROVIDER, buildProviderConfig(merged, streamSimple));
			})
			.catch((err) => {
				discoveryError = err instanceof Error ? err.message : String(err);
			});
	}

	registerAllTools(pi, config);

	pi.registerCommand("onnx", {
		description: "Show pi-onnx configuration",
		handler: async (_args, ctx) => {
			const lines: string[] = [];
			lines.push(`config: ${configPath()}`);
			lines.push(`cacheDir: ${config.cacheDir ?? "(HF default)"}`);
			lines.push(`device: ${config.device}   defaultDtype: ${config.defaultDtype}`);
			const discStatus = !config.discovery.enabled
				? "disabled"
				: discoveryError
					? `failed (${discoveryError})`
					: `enabled (limit=${config.discovery.limit}, pipelineTags=[${config.discovery.pipelineTags.join(", ")}])`;
			lines.push(`discovery: ${discStatus}`);
			lines.push(`pinned models:`);
			for (const m of config.models) {
				lines.push(`  - ${m.id}${m.name ? ` (${m.name})` : ""}  dtype=${m.dtype ?? config.defaultDtype}`);
			}
			lines.push(`discovered models: ${discovered.length}`);
			for (const m of discovered.slice(0, 20)) {
				lines.push(`  - ${m.id}`);
			}
			if (discovered.length > 20) {
				lines.push(`  ... and ${discovered.length - 20} more`);
			}
			lines.push("tools:");
			lines.push(`  embed:      ${config.tools.embed.enabled ? config.tools.embed.model : "disabled"}`);
			lines.push(`  classify:   ${config.tools.classify.enabled ? config.tools.classify.model : "disabled"}`);
			lines.push(`  transcribe: ${config.tools.transcribe.enabled ? config.tools.transcribe.model : "disabled"}`);
			ctx.ui.notify(lines.join("\n"));
		},
	});
}
