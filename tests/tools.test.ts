import { it } from "node:test";
import assert from "node:assert/strict";
import { ensureSupportedEmbedPooling } from "../src/tools.js";

it("rejects token-level embedding output shape requests", () => {
	assert.doesNotThrow(() => ensureSupportedEmbedPooling("mean"));
	assert.doesNotThrow(() => ensureSupportedEmbedPooling("cls"));
	assert.throws(
		() => ensureSupportedEmbedPooling("none"),
		/onnx_embed does not support tools\.embed\.pooling="none"/,
	);
});
