import React, {useMemo, useState} from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {pick, types, isErrorWithCode} from '@react-native-documents/picker';
import {INFERENCE_CONFIG} from './src/inferenceConfig';
import {mapErrorMessage, TranslateFailure, isBuildConfigReady} from './src/inferenceUx';

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

function App(): React.JSX.Element {
  const [selectedFile, setSelectedFile] = useState<{
    name: string;
    uri: string;
    type?: string | null;
    size?: number | null;
  } | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const configReady = isBuildConfigReady();
  const [status, setStatus] = useState(
    configReady
      ? 'Pick a <=5s video clip to translate.'
      : 'Build configuration missing. Set inferenceConfig.ts apiUrl/apiKey and rebuild.',
  );
  const [result, setResult] = useState<TranslateSuccess | null>(null);
  const [failure, setFailure] = useState<TranslateFailure | null>(null);

  const canRun = useMemo(
    () => Boolean(selectedFile?.uri) && configReady && !isUploading,
    [selectedFile?.uri, configReady, isUploading],
  );


  const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

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
      setStatus(
        configReady
          ? 'Video selected. Ready to upload.'
          : 'Build configuration missing. Set inferenceConfig.ts apiUrl/apiKey and rebuild.',
      );
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
    if (!configReady) {
      const misconfigured: TranslateFailure = {
        error_code: 'CONFIG_MISCONFIGURED',
        message: 'Build configuration is missing inference endpoint/key.',
        request_id: '',
        retryable: false,
        status: 'failed',
      };
      setFailure(misconfigured);
      setStatus(mapErrorMessage(misconfigured));
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

      const authHeaders = {
        'X-API-Key': INFERENCE_CONFIG.apiKey.trim(),
      };

      const submitResponse = await fetch(INFERENCE_CONFIG.apiUrl.trim(), {
        method: 'POST',
        headers: authHeaders,
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
        const pollUrl = pending.poll_url || `${INFERENCE_CONFIG.apiUrl.trim().replace(/\/$/, '')}/${pending.request_id}`;
        let attempts = 0;
        const maxAttempts = 20;

        while (attempts < maxAttempts) {
          attempts += 1;
          setStatus(`Processing… (attempt ${attempts}/${maxAttempts})`);
          await sleep(1000);

          const pollResponse = await fetch(pollUrl, {
            method: 'GET',
            headers: authHeaders,
          });
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

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Inference target</Text>
          <Text style={styles.cardValue}>{INFERENCE_CONFIG.targetLabel}</Text>
          <Text style={styles.muted}>{INFERENCE_CONFIG.apiUrl}</Text>
        </View>

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
