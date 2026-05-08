import SwiftUI
import Foundation
import Combine
import UniformTypeIdentifiers
import AVFoundation

private let confidenceThreshold = 0.70
private let latencySoftTargetMs = 500
private let artifactSchemaVersion = "ios_inference_v1"

@MainActor
final class InferenceViewModel: ObservableObject {
    @Published var selectedClip: DemoClip = .clip1
    @Published var selectedInputPath: InputPath = .tensor
    @Published var selectedRuntimeMode: RuntimeMode = defaultRuntimeMode
    @Published var strictProofMode: Bool = false

    @Published var primaryGlossText: String = "—"
    @Published var translationText: String = "—"
    @Published var rawGlossText: String = "—"
    @Published var confidenceText: String = "—"
    @Published var latencyText: String = "—"
    @Published var runtimeModeText: String = "—"
    @Published var routeReasonText: String = "—"
    @Published var requestIDText: String = "—"
    @Published var statusText: String = "Tap Run to start inference"
    @Published var isLoading: Bool = false
    @Published var modelPathDebugText: String = "Model Path: unresolved"

    private let client: LocalCactusInferenceProviding
    private let logger = InferenceArtifactLogger()

    init(client: LocalCactusInferenceProviding) {
        self.client = client
        self.modelPathDebugText = "Model Path: \(client.debugModelPathDescription())"
    }

    func runInference(uploadVideoData: Data? = nil, uploadFilename: String? = nil) {
        runInference(
            clip: selectedClip,
            inputPath: selectedInputPath,
            runtimeMode: selectedRuntimeMode,
            strictProofMode: strictProofMode,
            uploadVideoData: uploadVideoData,
            uploadFilename: uploadFilename
        )
    }

    func runRealProofInference() {
        runInference(
            clip: .clip1,
            inputPath: .tensor,
            runtimeMode: .cloud,
            strictProofMode: false,
            uploadVideoData: nil,
            uploadFilename: nil
        )
    }

    private func runInference(
        clip: DemoClip,
        inputPath: InputPath,
        runtimeMode: RuntimeMode,
        strictProofMode: Bool,
        uploadVideoData: Data?,
        uploadFilename: String?
    ) {
        isLoading = true
        statusText = "Uploading video and waiting for translation…"
        Task {
            modelPathDebugText = "Model Path: \(client.debugModelPathDescription())"
            let result = await client.infer(
                clip: clip,
                inputPath: inputPath,
                runtimeMode: runtimeMode,
                strictProofMode: strictProofMode,
                uploadVideoData: uploadVideoData,
                uploadFilename: uploadFilename
            )
            render(result: result)
            isLoading = false
            await logger.record(result: result, strictProofMode: strictProofMode)
        }
    }

    private func render(result: InferenceResult) {
        let isUnsure = result.confidence < confidenceThreshold
        primaryGlossText = isUnsure ? "UNSURE" : result.gloss
        rawGlossText = result.gloss.isEmpty ? "—" : result.gloss
        translationText = result.translation.isEmpty ? "—" : result.translation
        confidenceText = String(format: "%.1f%%", result.confidence * 100)
        latencyText = "\(result.latencyMs) ms"
        runtimeModeText = result.runtimeMode
        routeReasonText = result.routeReason
        requestIDText = result.requestID

        var pieces: [String] = [result.statusMessage]
        if result.latencyMs > latencySoftTargetMs {
            pieces.append("latency warning (>\(latencySoftTargetMs) ms)")
        }
        statusText = pieces.joined(separator: " · ")
    }
}

private struct LoggedInferenceRun: Codable {
    let schemaVersion: String
    let recordedAt: String
    let clip: String
    let inputPath: String
    let expectedGloss: String
    let predictedGloss: String
    let confidence: Double
    let latencyMs: Int
    let runtimeMode: String
    let routeReason: String
    let strictProofMode: Bool
    let success: Bool
}

private struct LoggedInferenceIndex: Codable {
    let schemaVersion: String
    var runs: [String]
}

private actor InferenceArtifactLogger {
    private let encoder: JSONEncoder = {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return enc
    }()

    func record(result: InferenceResult, strictProofMode: Bool) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }

        let run = LoggedInferenceRun(
            schemaVersion: artifactSchemaVersion,
            recordedAt: isoNow(),
            clip: result.clipID,
            inputPath: result.inputPath.rawValue,
            expectedGloss: result.expectedGloss,
            predictedGloss: result.gloss,
            confidence: result.confidence,
            latencyMs: result.latencyMs,
            runtimeMode: result.runtimeMode,
            routeReason: result.routeReason,
            strictProofMode: strictProofMode,
            success: result.success
        )

        let artifactsDir = docs.appendingPathComponent("ios_tracer_artifacts", isDirectory: true)
        let indexPath = artifactsDir.appendingPathComponent("session_index.json")
        try? FileManager.default.createDirectory(at: artifactsDir, withIntermediateDirectories: true)

        let timestampMs = Int(Date().timeIntervalSince1970 * 1000)
        let safeClip = result.clipID.replacingOccurrences(of: " ", with: "_").lowercased()
        let uniqueSuffix = UUID().uuidString.lowercased()
        let filename = "run_\(timestampMs)_\(safeClip)_\(result.inputPath.rawValue.lowercased())_\(uniqueSuffix).json"
        let runPath = artifactsDir.appendingPathComponent(filename)

        do {
            let runData = try encoder.encode(run)
            try runData.write(to: runPath)

            var index = loadIndex(from: indexPath)
            index.runs.append(filename)
            let indexData = try encoder.encode(index)
            try indexData.write(to: indexPath)
        } catch {
            // non-fatal logging errors for this tracer slice
        }
    }

    private func loadIndex(from path: URL) -> LoggedInferenceIndex {
        guard
            let data = try? Data(contentsOf: path),
            let decoded = try? JSONDecoder().decode(LoggedInferenceIndex.self, from: data)
        else {
            return LoggedInferenceIndex(schemaVersion: artifactSchemaVersion, runs: [])
        }
        return decoded
    }

    private func isoNow() -> String {
        ISO8601DateFormatter().string(from: Date())
    }
}

private var defaultRuntimeMode: RuntimeMode {
    .cloud
}

struct ContentView: View {
    @StateObject private var viewModel: InferenceViewModel
    @State private var showVideoImporter = false
    @State private var selectedVideoFilename: String = "No video selected"

    init(viewModel: InferenceViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("ASL Tracer Slice #1")
                .font(.title2)

            Picker("Runtime", selection: $viewModel.selectedRuntimeMode) {
                ForEach(RuntimeMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Picker("Input Path", selection: $viewModel.selectedInputPath) {
                ForEach(InputPath.allCases) { path in
                    Text(path.rawValue).tag(path)
                }
            }
            .pickerStyle(.segmented)

            Picker("Clip", selection: $viewModel.selectedClip) {
                ForEach(DemoClip.allCases) { clip in
                    Text(clip.rawValue).tag(clip)
                }
            }
            .pickerStyle(.segmented)

            Toggle("Strict Proof Mode (no fallback)", isOn: $viewModel.strictProofMode)
                .disabled(true)
                .opacity(0.5)

            HStack(spacing: 10) {
                Button("Select Video (<=5s)") {
                    showVideoImporter = true
                }
                .buttonStyle(.bordered)

                Text(selectedVideoFilename)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Button("Run Cloud Translation") {
                showVideoImporter = true
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.isLoading)

            Button("Real Proof Run") {
                viewModel.runRealProofInference()
            }
            .buttonStyle(.bordered)

            Group {
                Text("Primary Output: \(viewModel.primaryGlossText)")
                Text("Raw Top Gloss: \(viewModel.rawGlossText)")
                Text("Translation: \(viewModel.translationText)")
                Text("Confidence: \(viewModel.confidenceText)")
                Text("Latency: \(viewModel.latencyText)")
                Text("Runtime Mode: \(viewModel.runtimeModeText)")
                Text("Route Reason: \(viewModel.routeReasonText)")
                Text("Request ID: \(viewModel.requestIDText)")
                Text(viewModel.modelPathDebugText)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Text(viewModel.statusText)
                    .foregroundStyle(.secondary)
            }
            .font(.headline)

            Spacer()
        }
        .padding(20)
        .fileImporter(
            isPresented: $showVideoImporter,
            allowedContentTypes: [UTType.movie, UTType.mpeg4Movie, UTType.quickTimeMovie],
            allowsMultipleSelection: false
        ) { result in
            handleVideoSelection(result)
        }
    }

    private func handleVideoSelection(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            selectedVideoFilename = url.lastPathComponent
            do {
                let didStart = url.startAccessingSecurityScopedResource()
                defer {
                    if didStart { url.stopAccessingSecurityScopedResource() }
                }

                let asset = AVURLAsset(url: url)
                let seconds = CMTimeGetSeconds(asset.duration)
                if seconds.isFinite, seconds > 5.0 {
                    viewModel.statusText = String(format: "Selected clip is %.1fs (>5s). Please choose a shorter clip.", seconds)
                    return
                }

                let values = try url.resourceValues(forKeys: [.fileSizeKey])
                if let fileSize = values.fileSize, fileSize > 60_000_000 {
                    viewModel.statusText = "Selected clip appears too large for <=5s requirement. Please choose a shorter clip."
                    return
                }

                let data = try Data(contentsOf: url)
                viewModel.runInference(uploadVideoData: data, uploadFilename: url.lastPathComponent)
            } catch {
                viewModel.statusText = "Failed to read selected video: \(error.localizedDescription)"
            }
        case .failure(let error):
            viewModel.statusText = "Video selection canceled/failed: \(error.localizedDescription)"
        }
    }
}

#Preview {
    ContentView(viewModel: InferenceViewModel(client: LocalCactusInferenceClient()))
}
