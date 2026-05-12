import {INFERENCE_CONFIG} from './inferenceConfig';

export type TranslateFailure = {
  error_code: string;
  message: string;
  request_id: string;
  retryable: boolean;
  status?: 'failed';
};

export const isInferenceConfigReady = (config: {apiUrl: string; apiKey: string}) => {
  const url = config.apiUrl.trim();
  const key = config.apiKey.trim();
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return false;
  }
  if (!key || key === 'replace-with-asl-v1-api-key') {
    return false;
  }
  return true;
};

export const isBuildConfigReady = () => isInferenceConfigReady(INFERENCE_CONFIG);

export const mapErrorMessage = (failurePayload: TranslateFailure) => {
  const byCode: Record<string, string> = {
    UNAUTHORIZED: 'Authentication failed. Check your API key and try again.',
    RATE_LIMITED: 'Server is busy. Please retry in a moment.',
    PAYLOAD_TOO_LARGE: 'Video is too large. Upload a shorter clip (<=5 seconds).',
    VIDEO_DURATION_EXCEEDED: 'Video is too long. Upload a clip up to 5 seconds.',
    INVALID_VIDEO: 'Video could not be processed. Re-export and try again.',
    TIMEOUT: 'Processing timed out. Please retry.',
    UPSTREAM_FAILURE: 'Inference service is temporarily unavailable. Please retry.',
    INFERENCE_PROOF_MISSING:
      'Server proof verification failed (missing runtime proof). Contact support and include request ID.',
    INFERENCE_PROOF_INVALID:
      'Server proof verification failed (invalid runtime proof). Contact support and include request ID.',
    CLOUD_HANDOFF_FAILED:
      'Cloud handoff failed upstream. Please retry shortly; if it persists, contact support.',
    CONFIG_MISCONFIGURED:
      'App build is misconfigured. Ask support to set inference endpoint/key and rebuild the app.',
    NETWORK_ERROR: 'Could not connect to server. Check network and try again.',
  };

  if (byCode[failurePayload.error_code]) {
    return byCode[failurePayload.error_code];
  }

  if (failurePayload.retryable) {
    return `${failurePayload.message || 'Request failed.'} You can retry.`;
  }

  return failurePayload.message || 'Translation failed. Please try another clip.';
};
