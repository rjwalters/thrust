// Shared helper for resolving which Pong model JSON to load.
//
// Prefers the self-play artifact (`pong_self_play_model.json`) when it is
// present, and transparently falls back to the rule-based artifact
// (`pong_model.json`) otherwise. The raw JSON text is returned so callers can
// either feed it into the WASM policy loader (`load_policy_json`) or parse it
// for metadata display.

export type PongModelSource = "self-play" | "rule-based";

export interface PongModelResult {
	json: string;
	source: PongModelSource;
}

const SELF_PLAY_FILENAME = "pong_self_play_model.json";
const RULE_BASED_FILENAME = "pong_model.json";

export async function fetchPongModelJson(
	baseUrl: string,
): Promise<PongModelResult> {
	// Try self-play first.
	try {
		const response = await fetch(`${baseUrl}${SELF_PLAY_FILENAME}`);
		if (response.ok) {
			const json = await response.text();
			console.log(`Pong model loaded: ${SELF_PLAY_FILENAME} (self-play)`);
			return { json, source: "self-play" };
		}
		// Non-OK (e.g. 404) — fall through to rule-based.
		console.log(
			`No ${SELF_PLAY_FILENAME} (HTTP ${response.status}); falling back to ${RULE_BASED_FILENAME}`,
		);
	} catch (e) {
		console.warn(
			`Error fetching ${SELF_PLAY_FILENAME}; falling back to ${RULE_BASED_FILENAME}:`,
			e,
		);
	}

	// Fallback: rule-based artifact. Let the caller handle errors here so the
	// "no model file" path can still surface a heuristic agent.
	const response = await fetch(`${baseUrl}${RULE_BASED_FILENAME}`);
	if (!response.ok) {
		throw new Error(`HTTP ${response.status}`);
	}
	const json = await response.text();
	console.log(`Pong model loaded: ${RULE_BASED_FILENAME} (rule-based)`);
	return { json, source: "rule-based" };
}
