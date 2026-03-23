/**
 * Multimodal Fusion Store — manages voice + gesture intent fusion on the frontend.
 *
 * This store:
 *   1. Receives voice transcripts and gesture events from their respective components
 *   2. Sends them to the backend fusion engine via WebSocket
 *   3. Listens for fused intent notifications from the backend
 *   4. Provides reactive state for the UI to display fusion status
 *
 * Frontend WebSocket events:
 *   OUTBOUND (to backend):
 *     - voice_event: { transcript, confidence, is_final }
 *     - gesture_event: { gesture, confidence, data }
 *
 *   INBOUND (from backend notifications):
 *     - multimodal_intent: { command, voice_component, gesture_component, ... }
 */

import { writable, derived, get } from "svelte/store";

// ── Types ──

interface FusedIntent {
  command: string;
  voice_component: string;
  gesture_component: string;
  gesture_modifier: string;
  fusion_type: "single" | "voice_gesture" | "gesture_only" | "voice_only";
  confidence: number;
  timestamp: number;
  metadata: Record<string, any>;
}

interface FusionState {
  enabled: boolean;
  lastIntent: FusedIntent | null;
  recentIntents: FusedIntent[];
  pendingVoice: string | null;
  pendingGesture: string | null;
  fusionActive: boolean; // true when both modalities are active
}

// ── Store ──

const initialState: FusionState = {
  enabled: true,
  lastIntent: null,
  recentIntents: [],
  pendingVoice: null,
  pendingGesture: null,
  fusionActive: false,
};

function createMultimodalStore() {
  const { subscribe, set, update } = writable<FusionState>(initialState);

  return {
    subscribe,

    /** Enable/disable multimodal fusion */
    setEnabled(val: boolean) {
      update((s) => ({ ...s, enabled: val }));
    },

    /** Record a voice event (called from VoiceControl) */
    onVoiceInput(transcript: string, confidence: number, isFinal: boolean) {
      update((s) => ({
        ...s,
        pendingVoice: isFinal ? transcript : s.pendingVoice,
      }));
    },

    /** Record a gesture event (called from GestureControl) */
    onGestureInput(gesture: string, confidence: number) {
      update((s) => ({
        ...s,
        pendingGesture: gesture,
      }));
    },

    /** Handle a fused intent notification from the backend */
    onFusedIntent(intent: FusedIntent) {
      update((s) => ({
        ...s,
        lastIntent: intent,
        recentIntents: [...s.recentIntents.slice(-9), intent],
        pendingVoice: null,
        pendingGesture: null,
      }));
    },

    /** Update fusion active state (both voice + gesture controllers running) */
    setFusionActive(active: boolean) {
      update((s) => ({ ...s, fusionActive: active }));
    },

    /** Clear pending states */
    clearPending() {
      update((s) => ({
        ...s,
        pendingVoice: null,
        pendingGesture: null,
      }));
    },

    /** Reset the store */
    reset() {
      set(initialState);
    },
  };
}

export const multimodal = createMultimodalStore();

// ── Derived stores for easy UI binding ──

export const lastFusedIntent = derived(multimodal, ($m) => $m.lastIntent);
export const isFusionActive = derived(multimodal, ($m) => $m.fusionActive && $m.enabled);
export const hasPendingInput = derived(
  multimodal,
  ($m) => $m.pendingVoice !== null || $m.pendingGesture !== null
);
