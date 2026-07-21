/**
 * LiteLLM Provider Extension
 *
 * Fetches available models from LiteLLM's /v1/models endpoint, then
 * matches them against pi's built-in model metadata for Amazon Bedrock,
 * Azure OpenAI, Anthropic, OpenAI, and Google providers.
 *
 * Usage:
 *   LITELLM_BASE_URL=https://litellm.example.com LITELLM_API_KEY=sk-... pi -e ~/private/pi-extension-litellm
 *
 * On session start, pi shows a compact list of registered LiteLLM model IDs.
 * Use /litellm to inspect the full matched model metadata.
 * Then use /model to select any model under the "litellm" provider.
 */

import type { Api, Model } from "@earendil-works/pi-ai";
import { getBuiltinModels } from "@earendil-works/pi-ai/providers/all";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const DEFAULT_BASE_URL = "http://localhost:4000/v1";
const AZURE_PROVIDER = "azure-openai-responses";
const FALLBACK_PROVIDERS = ["anthropic", "openai", "google", "opencode", "opencode-go"] as const;
const LABEL = "📡 LiteLLM";

type PiModel = Model<Api>;
type LiteLLMModel = Omit<PiModel, "provider" | "baseUrl">;
type LiteLLMModelDebug = LiteLLMModel & { _source: string };

/**
 * Canonical slug for fuzzy ID matching across naming conventions:
 *   1. strip "provider." prefix  (e.g. "qwen."    from "qwen.qwen3-coder-480b-a35b-v1:0")
 *   2. strip "-vN…" version suffix (e.g. "-v1:0" from "qwen3-coder-480b-a35b-v1:0")
 *   3. collapse all separators → lowercase  ("qwen-3" === "qwen3")
 */
const slugify = (id: string) =>
	id.replace(/^[^.]+\./, "").replace(/-v\d.*$/i, "").replace(/[-._: /]/g, "").toLowerCase();

const formatModelDebug = (model: LiteLLMModelDebug) =>
	`LiteLLM model ${model.id} api: ${model.api} (${model._source}, reasoning=${model.reasoning})`;

const formatSpend = (spend: number) => Math.round(spend).toLocaleString();

const fetchAvailableIds = async (baseUrl: string, apiKey?: string): Promise<string[]> => {
	const headers: Record<string, string> = {};
	if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

	const res = await fetch(`${baseUrl}/models`, { headers });
	if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
	const data = (await res.json()) as { data: Array<{ id: string }> };
	return data.data.map((model) => model.id);
};

const buildModels = (availableIds: Set<string>): LiteLLMModelDebug[] => {
	// Strip `provider` and `baseUrl` — pi.registerProvider re-sets provider to
	// "litellm" and per-model baseUrl would override the LiteLLM proxy URL.
	const matched = new Set<string>();
	const models: LiteLLMModelDebug[] = [];

	const addModel = (litellmId: string, piModel: PiModel, source: string) => {
		if (matched.has(litellmId)) return;
		const { provider: _p, baseUrl: _b, ...rest } = piModel;
		matched.add(litellmId);
		models.push({
			...rest,
			id: litellmId,
			api:
				rest.api === "bedrock-converse-stream"
					? "openai-completions"
					: rest.api === "azure-openai-responses"
						? "openai-responses"
						: rest.api,
			_source: source,
		});
	};

	// Search Bedrock first. Slug normalisation handles prefix aliases
	// ("qwen." → "") and version suffixes ("-v1:0" → "") so that e.g.:
	//   LiteLLM "qwen-3-coder-480b-a35b"  ↔  pi "qwen.qwen3-coder-480b-a35b-v1:0"
	//   LiteLLM "minimax-m2"              ↔  pi "minimax.minimax-m2"
	const bedrockSlugMap = new Map<string, PiModel>();
	for (const model of getBuiltinModels("amazon-bedrock")) {
		bedrockSlugMap.set(slugify(model.id), model);
	}
	for (const litellmId of availableIds) {
		const piModel = bedrockSlugMap.get(slugify(litellmId));
		if (piModel) addModel(litellmId, piModel, `bedrock-fuzzy(${piModel.id})`);
	}

	// Then exact-match Azure OpenAI before the remaining providers.
	for (const model of getBuiltinModels(AZURE_PROVIDER)) {
		if (availableIds.has(model.id)) addModel(model.id, model, AZURE_PROVIDER);
	}

	// Finally exact-match the rest of the built-in metadata providers.
	for (const provider of FALLBACK_PROVIDERS) {
		for (const model of getBuiltinModels(provider)) {
			if (availableIds.has(model.id)) addModel(model.id, model, provider);
		}
	}

	return models;
};

export default async function (pi: ExtensionAPI) {
	const baseUrl = process.env.LITELLM_BASE_URL || DEFAULT_BASE_URL;
	const apiKey = process.env.LITELLM_API_KEY;
	const state = {
		availableIds: [] as string[],
		models: [] as LiteLLMModelDebug[],
		error: undefined as string | undefined,
		keySpend: undefined as number | undefined,
	};

	const updateStatus = (ctx: ExtensionContext) => {
		if (state.error) {
			ctx.ui.setStatus("litellm", ctx.ui.theme.fg("error", `${LABEL}: error`));
			return;
		}
		if (state.models.length === 0) {
			ctx.ui.setStatus("litellm", ctx.ui.theme.fg("error", `${LABEL}: 0 models`));
			return;
		}
		let summary = `${LABEL}: ${state.models.length}/${state.availableIds.length} models`;
		if (state.keySpend !== undefined) {
			summary += `, $${formatSpend(state.keySpend)} key spend`;
		}
		ctx.ui.setStatus("litellm", ctx.ui.theme.fg("accent", summary));
	};

	pi.registerCommand("litellm", {
		description: "Show LiteLLM models discovered from the configured proxy",
		handler: async (_args, ctx) => {
			if (state.error) {
				ctx.ui.notify(state.error, "error");
				return;
			}
			if (state.models.length === 0) {
				ctx.ui.notify("LiteLLM: no models loaded", "warning");
				return;
			}

			const matchedIds = new Set(state.models.map((model) => model.id));
			const unmatchedIds = state.availableIds.filter((id) => !matchedIds.has(id)).sort();

			const lines = [
				`LiteLLM models (${state.models.length}/${state.availableIds.length} matched):`,
				...state.models.map(formatModelDebug),
			];
			if (unmatchedIds.length > 0) {
				lines.push("", `Unmatched (${unmatchedIds.length}):`, ...unmatchedIds.map((id) => `  ${id}`));
			}

			ctx.ui.notify(lines.join("\n"), "info");
		},
	});

	pi.on("after_provider_response", (event, ctx) => {
		if (ctx.model?.provider !== "litellm") return;
		const value = event.headers["x-litellm-key-spend"];
		if (value !== undefined) {
			const spend = Number(value);
			if (Number.isFinite(spend)) state.keySpend = spend;
		}
		updateStatus(ctx);
	});

	pi.on("session_start", (_event, ctx) => {
		// Fire-and-forget: pi awaits session_start handlers, then synchronously
		// runs showLoadedResources / showStartupNoticesIfNeeded /
		// renderInitialMessages, which dump content into the chat container
		// after our notify. An inline `await` doesn't help — it just suspends
		// pi's emit chain and the post-emit sync code still runs after we
		// resume, so the notify lands at the top and scrolls off the viewport.
		// setTimeout(0) runs the notify after pi's sync startup rendering is
		// done, so it lands at the bottom of the chat, above the editor.
		setTimeout(() => {
			if (state.error) {
				ctx.ui.notify(state.error, "error");
				updateStatus(ctx);
				return;
			}
			if (state.models.length === 0) {
				updateStatus(ctx);
				return;
			}
			ctx.ui.notify(`LiteLLM: ${state.models.map((model) => model.id).join(", ")}`, "info");
			updateStatus(ctx);
		}, 0);
	});

	try {
		state.availableIds = await fetchAvailableIds(baseUrl, apiKey);
	} catch (e) {
		const message = e instanceof Error ? e.message : String(e);
		state.error = `LiteLLM: failed to fetch models from ${baseUrl}/models (${message})`;
		console.error(`Failed to fetch LiteLLM models from ${baseUrl}/models:`, e);
		return;
	}

	state.models = buildModels(new Set(state.availableIds));
	if (state.models.length === 0) {
		state.error = `LiteLLM: no models matched built-in metadata. Available: ${state.availableIds.join(", ")}`;
		console.error(state.error);
		return;
	}

	pi.registerProvider("litellm", {
		baseUrl,
		apiKey: "$LITELLM_API_KEY",
		models: state.models.map(({ _source: _source, ...model }) => model),
	});
}
