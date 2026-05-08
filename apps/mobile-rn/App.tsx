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
};

type TranslateFailure = {
  error_code: string;
  message: string;
  request_id: string;
  retryable: boolean;
};

const DEFAULT_ENDPOINT = 'http://192.168.1.42:8000/v1/translate-sign';

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

      const response = await fetch(endpoint.trim(), {
        method: 'POST',
        body: form,
      });

      const json = (await response.json()) as TranslateSuccess | TranslateFailure;
      if (response.ok) {
        const ok = json as TranslateSuccess;
        setResult(ok);
        setStatus(`Translation complete (${ok.latency_ms} ms).`);
      } else {
        const err = json as TranslateFailure;
        setFailure(err);
        setStatus(err.message || 'Translation failed.');
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Could not connect to the server.');
      setFailure({
        error_code: 'NETWORK_ERROR',
        message: error instanceof Error ? error.message : 'Could not connect to the server.',
        request_id: '',
        retryable: true,
      });
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

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Selected clip</Text>
          <Text style={styles.cardValue}>{selectedFile?.name ?? 'None'}</Text>
          <Text style={styles.muted}>URI: {selectedFile?.uri ?? '-'}</Text>
          <Text style={styles.muted}>Size: {selectedFile?.size ?? '-'} bytes</Text>
        </View>

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
            <Text style={styles.muted}>Request ID: {result.request_id}</Text>
          </View>
        ) : null}

        {failure ? (
          <View style={styles.errorCard}>
            <Text style={styles.cardTitle}>Error</Text>
            <Text style={styles.errorText}>{failure.error_code}: {failure.message}</Text>
            <Text style={styles.muted}>Request ID: {failure.request_id || '-'}</Text>
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
