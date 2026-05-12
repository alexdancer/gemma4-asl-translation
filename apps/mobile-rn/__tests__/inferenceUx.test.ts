jest.mock('../src/inferenceConfig', () => ({
  INFERENCE_CONFIG: {
    apiUrl: 'http://127.0.0.1:8000/v1/translate-sign',
    apiKey: 'test-key',
  },
}));

import {isBuildConfigReady, isInferenceConfigReady, mapErrorMessage} from '../src/inferenceUx';

describe('inferenceUx', () => {
  test('build config is ready with valid http url and non-placeholder key', () => {
    expect(isBuildConfigReady()).toBe(true);
  });

  test('maps proof failures to explicit actionable messages', () => {
    const msgMissing = mapErrorMessage({
      error_code: 'INFERENCE_PROOF_MISSING',
      message: 'missing proof',
      request_id: 'r1',
      retryable: false,
    });
    expect(msgMissing).toContain('proof verification failed');

    const msgInvalid = mapErrorMessage({
      error_code: 'INFERENCE_PROOF_INVALID',
      message: 'invalid proof',
      request_id: 'r2',
      retryable: false,
    });
    expect(msgInvalid).toContain('proof verification failed');
  });

  test('maps cloud handoff failure to retry guidance', () => {
    const message = mapErrorMessage({
      error_code: 'CLOUD_HANDOFF_FAILED',
      message: 'handoff failed',
      request_id: 'r3',
      retryable: true,
    });
    expect(message).toContain('Cloud handoff failed upstream');
  });

  test('build config is not ready with placeholder key', () => {
    expect(
      isInferenceConfigReady({
        apiUrl: 'http://127.0.0.1:8000/v1/translate-sign',
        apiKey: 'replace-with-asl-v1-api-key',
      }),
    ).toBe(false);
  });

  test('build config is not ready with non-http url', () => {
    expect(
      isInferenceConfigReady({
        apiUrl: '127.0.0.1:8000/v1/translate-sign',
        apiKey: 'real-key',
      }),
    ).toBe(false);
  });
});
