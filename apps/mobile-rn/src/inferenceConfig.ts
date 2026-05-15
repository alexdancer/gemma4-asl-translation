import {NativeModules} from 'react-native';
import {LOCAL_INFERENCE_TARGET} from './inferenceLocal.generated';

export type InferenceBuildConfig = {
  apiUrl: string;
  apiKey: string;
  targetLabel: string;
};

export const DEFAULT_SIMULATOR_API_URL = 'http://127.0.0.1:8000/v1/translate-sign';
export const DEFAULT_API_KEY = 'dev-local-key-1';

export const apiUrlFromMetroScriptUrl = (scriptUrl?: string | null) => {
  if (!scriptUrl) {
    return DEFAULT_SIMULATOR_API_URL;
  }

  try {
    const metroUrl = new URL(scriptUrl);
    if (!metroUrl.hostname) {
      return DEFAULT_SIMULATOR_API_URL;
    }
    const apiHost = metroUrl.hostname === 'localhost' ? '127.0.0.1' : metroUrl.hostname;
    return `http://${apiHost}:8000/v1/translate-sign`;
  } catch {
    return DEFAULT_SIMULATOR_API_URL;
  }
};

export const targetLabelFromApiUrl = (apiUrl: string) => {
  if (apiUrl.includes('127.0.0.1') || apiUrl.includes('localhost')) {
    return 'local simulator backend';
  }
  return 'local physical-device backend';
};

const sourceCodeScriptUrl = NativeModules.SourceCode?.scriptURL as string | undefined;
const generatedApiUrl = LOCAL_INFERENCE_TARGET.apiUrl?.trim();
const resolvedApiUrl = generatedApiUrl || (__DEV__ ? apiUrlFromMetroScriptUrl(sourceCodeScriptUrl) : DEFAULT_SIMULATOR_API_URL);
const resolvedApiKey = LOCAL_INFERENCE_TARGET.apiKey?.trim() || DEFAULT_API_KEY;

// Development inference wiring.
// scripts/dev/run_mobile_stack.sh writes inferenceLocal.generated.ts before launch:
// - simulator mode uses http://127.0.0.1:8000/v1/translate-sign
// - physical-device mode uses http://<mac-lan-ip>:8000/v1/translate-sign
// The app calls only the local API. Provider/HF/Cactus secrets stay server-side.
export const INFERENCE_CONFIG: InferenceBuildConfig = {
  apiUrl: resolvedApiUrl,
  apiKey: resolvedApiKey,
  targetLabel: targetLabelFromApiUrl(resolvedApiUrl),
};
