// SPDX-License-Identifier: MIT
// Copyright (C) Jarkko Sakkinen 2026

import { HF_PREFIX, type PipelineTag } from "./config.js";

export interface DiscoveredModel {
	id: string;
	name: string;
}

export interface DiscoveryOptions {
	limit: number;
	pipelineTags: PipelineTag[];
	signal?: AbortSignal;
}

interface HubModel {
	id?: string;
	modelId?: string;
}

async function fetchByPipelineTag(
	tag: PipelineTag,
	limit: number,
	signal: AbortSignal,
): Promise<HubModel[]> {
	const url = new URL("https://huggingface.co/api/models");
	url.searchParams.set("author", "onnx-community");
	url.searchParams.set("pipeline_tag", tag);
	url.searchParams.set("sort", "downloads");
	url.searchParams.set("direction", "-1");
	url.searchParams.set("limit", String(limit));

	const res = await fetch(url, {
		signal,
		headers: { "User-Agent": "pi-provider-onnx-community" },
	});
	if (!res.ok) throw new Error(`HF Hub ${res.status} ${res.statusText}`);
	return (await res.json()) as HubModel[];
}

export async function discoverOnnxCommunityModels(opts: DiscoveryOptions): Promise<DiscoveredModel[]> {
	const signal = opts.signal ?? AbortSignal.timeout(10_000);
	const results = await Promise.all(
		opts.pipelineTags.map((tag) => fetchByPipelineTag(tag, opts.limit, signal)),
	);

	const seen = new Set<string>();
	const out: DiscoveredModel[] = [];
	for (const list of results) {
		for (const m of list) {
			const id = m.id ?? m.modelId;
			if (!id || !id.startsWith(HF_PREFIX)) continue;
			if (seen.has(id)) continue;
			seen.add(id);
			out.push({ id, name: id.slice(HF_PREFIX.length) });
		}
	}
	return out;
}
