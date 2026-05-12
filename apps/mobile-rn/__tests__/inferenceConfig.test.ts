import {
  apiUrlFromMetroScriptUrl,
  DEFAULT_SIMULATOR_API_URL,
  INFERENCE_CONFIG,
  targetLabelFromApiUrl,
} from '../src/inferenceConfig';

describe('inferenceConfig', () => {
  test('uses generated default simulator config when no dev runner has switched modes', () => {
    expect(INFERENCE_CONFIG.apiUrl).toBe(DEFAULT_SIMULATOR_API_URL);
    expect(INFERENCE_CONFIG.apiKey).toBe('dev-local-key-1');
    expect(INFERENCE_CONFIG.targetLabel).toBe('local simulator backend');
  });

  test('uses simulator localhost fallback when Metro script URL is unavailable', () => {
    expect(apiUrlFromMetroScriptUrl(undefined)).toBe(DEFAULT_SIMULATOR_API_URL);
    expect(apiUrlFromMetroScriptUrl('not a url')).toBe(DEFAULT_SIMULATOR_API_URL);
  });

  test('maps simulator Metro URL to local FastAPI endpoint', () => {
    expect(apiUrlFromMetroScriptUrl('http://127.0.0.1:8081/index.bundle?platform=ios')).toBe(
      'http://127.0.0.1:8000/v1/translate-sign',
    );
    expect(apiUrlFromMetroScriptUrl('http://localhost:8081/index.bundle?platform=ios')).toBe(
      'http://127.0.0.1:8000/v1/translate-sign',
    );
  });

  test('maps physical iPhone Metro URL to Mac LAN FastAPI endpoint', () => {
    expect(apiUrlFromMetroScriptUrl('http://192.168.68.69:8081/index.bundle?platform=ios')).toBe(
      'http://192.168.68.69:8000/v1/translate-sign',
    );
  });

  test('labels simulator and physical-device endpoints', () => {
    expect(targetLabelFromApiUrl('http://127.0.0.1:8000/v1/translate-sign')).toBe(
      'local simulator backend',
    );
    expect(targetLabelFromApiUrl('http://192.168.68.69:8000/v1/translate-sign')).toBe(
      'local physical-device backend',
    );
  });
});
