import React, {useMemo, useState} from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {pick, types, isErrorWithCode} from '@react-native-documents/picker';

type TranslateSuccess = {
  request_id: string;
  gloss: string;
  translation: string;
  confidence: number;
  latency_ms: number;
  status?: 'completed';
};

type TranslatePending = {
  request_id: string;
  status: 'queued' | 'processing';
  poll_url?: string;
};

type TranslateFailure = {
  error_code: string;
  message: string;
  request_id: string;
  retryable: boolean;
  status?: 'failed';
};

const DEFAULT_ENDPOINT = '';

function App(): React.JSX.Element {
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINT);
  const [selectedFile, setSelectedFile] = useState<{
    name: string;
    uri: string;
    type?: string | null;
    size?: number | null;
  } | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState('Pick a <=5s video clip to translate.');
  const [result, setResult] = useState<TranslateSuccess | null>(null);
  const [failure, setFailure] = useState<TranslateFailure | null>(null);

  const canRun = useMemo(
    () => Boolean(selectedFile?.uri) && Boolean(endpoint.trim()) && !isUploading,
    [selectedFile?.uri, endpoint, isUploading],
  );

  const mapErrorMessage = (failurePayload: TranslateFailure) => {
    const byCode: Record<string, string> = {
      UNAUTHORIZED: 'Authentication failed. Check your API key and try again.',
      RATE_LIMITED: 'Server is busy. Please retry in a moment.',
      PAYLOAD_TOO_LARGE: 'Video is too large. Upload a shorter clip (<=5 seconds).',
      VIDEO_DURATION_EXCEEDED: 'Video is too long. Upload a clip up to 5 seconds.',
      INVALID_VIDEO: 'Video could not be processed. Re-export and try again.',
      TIMEOUT: 'Processing timed out. Please retry.',
      UPSTREAM_FAILURE: 'Inference service is temporarily unavailable. Please retry.',
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

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  const onPickVideo = async () => {
    try {
      const [doc] = await pick({
        type: [types.video],
        mode: 'import',
        allowMultiSelection: false,
      });
      if (!doc?.uri) {
        return;
      }
      setSelectedFile({
        name: doc.name ?? 'clip.mov',
        uri: doc.uri,
        type: doc.type,
        size: doc.size,
      });
      setResult(null);
      setFailure(null);
      setStatus('Video selected. Ready to upload.');
    } catch (error) {
      if (isErrorWithCode(error) && error.code === 'OPERATION_CANCELED') {
        return;
      }
      Alert.alert('Pick failed', error instanceof Error ? error.message : 'Unknown picker error');
    }
  };

  const onRunCloudTranslation = async () => {
    if (!selectedFile?.uri) {
      return;
    }
    setIsUploading(true);
    setResult(null);
    setFailure(null);
    setStatus('Uploading video and waiting for translation…');

    try {
      const form = new FormData();
      form.append('video', {
        uri: selectedFile.uri,
        name: selectedFile.name,
        type: selectedFile.type ?? 'video/quicktime',
      } as unknown as Blob);

      const submitResponse = await fetch(endpoint.trim(), {
        method: 'POST',
        body: form,
      });

      const submitJson = (await submitResponse.json()) as TranslateSuccess | TranslateFailure | TranslatePending;

      if (submitResponse.ok && (submitJson as TranslateSuccess).translation) {
        const ok = submitJson as TranslateSuccess;
        setResult(ok);
        setStatus(`Translation complete (${ok.latency_ms} ms).`);
        return;
      }

      if (submitResponse.status === 202 || (submitJson as TranslatePending).status === 'queued' || (submitJson as TranslatePending).status === 'processing') {
        const pending = submitJson as TranslatePending;
        const pollUrl = pending.poll_url || `${endpoint.trim().replace(/\/$/, '')}/${pending.request_id}`;
        let attempts = 0;
        const maxAttempts = 20;

        while (attempts < maxAttempts) {
          attempts += 1;
          setStatus(`Processing… (attempt ${attempts}/${maxAttempts})`);
          await sleep(1000);

          const pollResponse = await fetch(pollUrl, {method: 'GET'});
          const pollJson = (await pollResponse.json()) as TranslateSuccess | TranslateFailure | TranslatePending;

          if (pollResponse.ok && (pollJson as TranslateSuccess).translation) {
            const ok = pollJson as TranslateSuccess;
            setResult(ok);
            setStatus(`Translation complete (${ok.latency_ms} ms).`);
            return;
          }

          if ((pollJson as TranslatePending).status === 'queued' || (pollJson as TranslatePending).status === 'processing') {
            continue;
          }

          const err = pollJson as TranslateFailure;
          setFailure(err);
          setStatus(mapErrorMessage(err));
          return;
        }

        const timeoutFailure: TranslateFailure = {
          error_code: 'TIMEOUT',
          message: 'Processing did not complete in time.',
          request_id: pending.request_id,
          retryable: true,
          status: 'failed',
        };
        setFailure(timeoutFailure);
        setStatus(mapErrorMessage(timeoutFailure));
        return;
      }

      const err = submitJson as TranslateFailure;
      setFailure(err);
      setStatus(mapErrorMessage(err));
    } catch (error) {
      const networkFailure: TranslateFailure = {
        error_code: 'NETWORK_ERROR',
        message: error instanceof Error ? error.message : 'Could not connect to the server.',
        request_id: '',
        retryable: true,
        status: 'failed',
      };
      setFailure(networkFailure);
      setStatus(mapErrorMessage(networkFailure));
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>ASL v2 Cloud Translation (React Native)</Text>

        <Text style={styles.label}>Cloud endpoint</Text>
        <TextInput
          value={endpoint}
          onChangeText={setEndpoint}
          autoCapitalize="none"
          autoCorrect={false}
          style={styles.input}
          placeholder="https://your-domain/v1/translate-sign"
          placeholderTextColor="#94a3b8"
        />

        <TouchableOpacity style={styles.secondaryButton} onPress={onPickVideo}>
          <Text style={styles.buttonText}>Select Video</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.primaryButton, !canRun && styles.disabledButton]}
          onPress={onRunCloudTranslation}
          disabled={!canRun}>
          {isUploading ? <ActivityIndicator color="#0f172a" /> : <Text style={styles.runButtonText}>Run Cloud Translation</Text>}
        </TouchableOpacity>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Status</Text>
          <Text style={styles.cardValue}>{status}</Text>
        </View>

        {result ? (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Result</Text>
            <Text style={styles.cardValue}>Gloss: {result.gloss}</Text>
            <Text style={styles.cardValue}>Translation: {result.translation}</Text>
            <Text style={styles.muted}>Confidence: {result.confidence}</Text>
          </View>
        ) : null}

        {failure ? (
          <View style={styles.errorCard}>
            <Text style={styles.cardTitle}>Error</Text>
            <Text style={styles.errorText}>{failure.error_code}: {mapErrorMessage(failure)}</Text>
            <Text style={styles.muted}>Retryable: {failure.retryable ? 'Yes' : 'No'}</Text>
          </View>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {flex: 1, backgroundColor: '#020617'},
  container: {padding: 20, gap: 12},
  title: {fontSize: 22, fontWeight: '700', color: '#e2e8f0', marginBottom: 10},
  label: {fontSize: 14, color: '#cbd5e1', marginBottom: 2},
  input: {
    borderWidth: 1,
    borderColor: '#334155',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: '#e2e8f0',
    backgroundColor: '#0f172a',
  },
  primaryButton: {
    backgroundColor: '#22d3ee',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
  },
  secondaryButton: {
    backgroundColor: '#1e293b',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  disabledButton: {opacity: 0.5},
  buttonText: {fontWeight: '600', color: '#e2e8f0'},
  runButtonText: {fontWeight: '700', color: '#0f172a'},
  card: {
    backgroundColor: '#0f172a',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: '#1e293b',
  },
  errorCard: {
    backgroundColor: '#2b1320',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: '#7f1d1d',
  },
  cardTitle: {color: '#93c5fd', fontWeight: '600', marginBottom: 6},
  cardValue: {color: '#e2e8f0', marginBottom: 2},
  muted: {color: '#94a3b8', fontSize: 12},
  errorText: {color: '#fecaca'},
});

export default App;
