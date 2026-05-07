import Foundation

#if canImport(Cactus)
import Cactus
#endif

struct InferenceResult {
    let clipID: String
    let inputPath: InputPath
    let gloss: String
    let confidence: Double
    let latencyMs: Int
    let runtimeMode: String
    let routeReason: String
    let statusMessage: String
    let expectedGloss: String
    let success: Bool
}

enum InputPath: String, CaseIterable, Identifiable {
    case tensor = "Tensor"
    case video = "Video"

    var id: String { rawValue }
}

enum DemoClip: String, CaseIterable, Identifiable {
    case clip1 = "Clip 1"
    case clip2 = "Clip 2"
    case clip3 = "Clip 3"

    var id: String { rawValue }

    var fixtureKey: String {
        switch self {
        case .clip1: return "clip_1"
        case .clip2: return "clip_2"
        case .clip3: return "clip_3"
        }
    }
}

enum RuntimeMode: String, CaseIterable, Identifiable {
    case demo = "Demo"
    case realLocal = "RealLocal"

    var id: String { rawValue }
}

protocol LocalCactusInferenceProviding {
    func infer(clip: DemoClip, inputPath: InputPath, runtimeMode: RuntimeMode, strictProofMode: Bool) async -> InferenceResult
    func debugModelPathDescription() -> String
}

private struct FixtureClipPayload: Decodable {
    let expectedGloss: String
    let confidence: Double
}

private struct FixtureBundlePayload: Decodable {
    let schemaVersion: String
    let tensor: [String: FixtureClipPayload]
    let video: [String: FixtureClipPayload]
}

enum FixtureLoadError: Error {
    case missingResource
    case decodeFailed
    case missingClipEntry
}

struct LocalCactusPrediction {
    let gloss: String
    let confidence: Double
}

enum CactusRuntimeError: Error {
    case sdkUnavailable
    case modelPathUnavailable
    case modelInitFailed
    case responseDecodeFailed
    case invocationFailed
}

protocol CactusRuntimeProviding {
    func infer(clip: DemoClip, inputPath: InputPath) async throws -> LocalCactusPrediction
    func debugResolvedModelPath() -> String?
}

private struct CactusCompletionResponse: Decodable {
    let success: Bool
    let response: String?
    let confidence: Double?
}

final class CactusRuntimeAdapter: CactusRuntimeProviding {
    #if canImport(Cactus)
    private var model: CactusModelT?
    #endif

    func debugResolvedModelPath() -> String? {
        #if canImport(Cactus)
        return resolveModelPath()
        #else
        return nil
        #endif
    }

    func infer(clip: DemoClip, inputPath: InputPath) async throws -> LocalCactusPrediction {
        #if canImport(Cactus)
        let modelHandle = try loadModelIfNeeded()
        let messages = try makeMessagesJSON(clip: clip, inputPath: inputPath)
        let options = "{\"max_tokens\":64,\"temperature\":0.0}"

        let responseJson: String
        do {
            responseJson = try cactusComplete(modelHandle, messages, options, nil, nil)
        } catch {
            throw CactusRuntimeError.invocationFailed
        }

        guard let data = responseJson.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(CactusCompletionResponse.self, from: data)
        else {
            throw CactusRuntimeError.responseDecodeFailed
        }

        guard decoded.success else {
            throw CactusRuntimeError.invocationFailed
        }

        let raw = (decoded.response ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard raw.isEmpty == false else {
            throw CactusRuntimeError.invocationFailed
        }

        return LocalCactusPrediction(
            gloss: raw,
            confidence: decoded.confidence ?? 0.0
        )
        #else
        _ = (clip, inputPath)
        throw CactusRuntimeError.sdkUnavailable
        #endif
    }

    #if canImport(Cactus)
    deinit {
        if let model {
            cactusDestroy(model)
        }
    }

    private func loadModelIfNeeded() throws -> CactusModelT {
        if let model {
            return model
        }

        guard let modelPath = resolveModelPath() else {
            throw CactusRuntimeError.modelPathUnavailable
        }

        guard let initialized = try? cactusInit(modelPath, nil, false) else {
            throw CactusRuntimeError.modelInitFailed
        }

        model = initialized
        return initialized
    }

    private func resolveModelPath() -> String? {
        if let explicit = ProcessInfo.processInfo.environment["CACTUS_MODEL_PATH"], explicit.isEmpty == false {
            return explicit
        }

        if let bundled = Bundle.main.path(forResource: "cactus-model", ofType: nil) {
            return bundled
        }

        return nil
    }

    private func makeMessagesJSON(clip: DemoClip, inputPath: InputPath) throws -> String {
        let prompt = "Return exactly one ASL gloss label for \(clip.rawValue) using \(inputPath.rawValue) path. Output only the label token."
        let messages: [[String: String]] = [["role": "user", "content": prompt]]
        let data = try JSONSerialization.data(withJSONObject: messages)
        return String(decoding: data, as: UTF8.self)
    }
    #endif
}

final class LocalCactusInferenceClient: LocalCactusInferenceProviding {
    private let runtimeAdapter: CactusRuntimeProviding

    init(runtimeAdapter: CactusRuntimeProviding = CactusRuntimeAdapter()) {
        self.runtimeAdapter = runtimeAdapter
    }

    func debugModelPathDescription() -> String {
        #if canImport(Cactus)
        return runtimeAdapter.debugResolvedModelPath() ?? "unresolved"
        #else
        return "sdk_unavailable"
        #endif
    }

    func infer(clip: DemoClip, inputPath: InputPath, runtimeMode: RuntimeMode, strictProofMode: Bool) async -> InferenceResult {
        let start = Date()

        switch runtimeMode {
        case .demo:
            do {
                let fixture = try loadFixture(clip: clip, inputPath: inputPath)
                let latency = max(Int(Date().timeIntervalSince(start) * 1000), 1)
                return InferenceResult(
                    clipID: clip.rawValue,
                    inputPath: inputPath,
                    gloss: fixture.expectedGloss,
                    confidence: fixture.confidence,
                    latencyMs: latency,
                    runtimeMode: "fixture",
                    routeReason: "fallback_fixture_demo_mode",
                    statusMessage: "Demo mode: fixture inference complete",
                    expectedGloss: fixture.expectedGloss,
                    success: true
                )
            } catch {
                let latency = max(Int(Date().timeIntervalSince(start) * 1000), 1)
                return InferenceResult(
                    clipID: clip.rawValue,
                    inputPath: inputPath,
                    gloss: "",
                    confidence: 0,
                    latencyMs: latency,
                    runtimeMode: "fixture",
                    routeReason: "fixture_load_failed",
                    statusMessage: "Demo mode failed: fixture load error",
                    expectedGloss: "",
                    success: false
                )
            }

        case .realLocal:
            var lastError: CactusRuntimeError = .invocationFailed
            for _ in 1...2 {
                do {
                    let prediction = try await runtimeAdapter.infer(clip: clip, inputPath: inputPath)
                    let latency = max(Int(Date().timeIntervalSince(start) * 1000), 1)
                    return InferenceResult(
                        clipID: clip.rawValue,
                        inputPath: inputPath,
                        gloss: prediction.gloss,
                        confidence: prediction.confidence,
                        latencyMs: latency,
                        runtimeMode: "local_cactus",
                        routeReason: "local_cactus_runtime_success",
                        statusMessage: "RealLocal: local Cactus runtime inference complete",
                        expectedGloss: (try? expectedGlossFor(clip: clip, inputPath: inputPath)) ?? "",
                        success: true
                    )
                } catch let error as CactusRuntimeError {
                    lastError = error
                } catch {
                    lastError = .invocationFailed
                }
            }

            if strictProofMode {
                let latency = max(Int(Date().timeIntervalSince(start) * 1000), 1)
                let routeReason: String
                let statusMessage: String
                switch lastError {
                case .sdkUnavailable:
                    routeReason = "strict_proof_sdk_unavailable"
                    statusMessage = "Strict proof mode: Cactus SDK unavailable (no fallback)"
                case .modelPathUnavailable:
                    routeReason = "strict_proof_model_path_unavailable"
                    statusMessage = "Strict proof mode: model path unavailable (set CACTUS_MODEL_PATH or bundle cactus-model)"
                case .modelInitFailed:
                    routeReason = "strict_proof_model_init_failed"
                    statusMessage = "Strict proof mode: cactusInit failed (invalid/incompatible model bundle)"
                case .responseDecodeFailed:
                    routeReason = "strict_proof_response_decode_failed"
                    statusMessage = "Strict proof mode: runtime response decode failed"
                case .invocationFailed:
                    routeReason = "strict_proof_local_runtime_failed"
                    statusMessage = "Strict proof mode: local runtime failed after retries (no fallback)"
                }

                return InferenceResult(
                    clipID: clip.rawValue,
                    inputPath: inputPath,
                    gloss: "",
                    confidence: 0,
                    latencyMs: latency,
                    runtimeMode: "local_cactus",
                    routeReason: routeReason,
                    statusMessage: statusMessage,
                    expectedGloss: (try? expectedGlossFor(clip: clip, inputPath: inputPath)) ?? "",
                    success: false
                )
            }

            do {
                let fixture = try loadFixture(clip: clip, inputPath: inputPath)
                let latency = max(Int(Date().timeIntervalSince(start) * 1000), 1)
                return InferenceResult(
                    clipID: clip.rawValue,
                    inputPath: inputPath,
                    gloss: fixture.expectedGloss,
                    confidence: fixture.confidence,
                    latencyMs: latency,
                    runtimeMode: "fixture",
                    routeReason: "fallback_after_local_runtime_failure",
                    statusMessage: "RealLocal failed; using fixture fallback",
                    expectedGloss: fixture.expectedGloss,
                    success: true
                )
            } catch {
                let latency = max(Int(Date().timeIntervalSince(start) * 1000), 1)
                return InferenceResult(
                    clipID: clip.rawValue,
                    inputPath: inputPath,
                    gloss: "",
                    confidence: 0,
                    latencyMs: latency,
                    runtimeMode: "fixture",
                    routeReason: "fixture_load_failed_after_local_runtime_failure",
                    statusMessage: "RealLocal failed and fixture fallback unavailable",
                    expectedGloss: "",
                    success: false
                )
            }
        }
    }

    private func expectedGlossFor(clip: DemoClip, inputPath: InputPath) throws -> String {
        let fixture = try loadFixture(clip: clip, inputPath: inputPath)
        return fixture.expectedGloss
    }

    private func loadFixture(clip: DemoClip, inputPath: InputPath) throws -> FixtureClipPayload {
        guard let url = Bundle.main.url(forResource: "local_cactus_clips", withExtension: "json") else {
            throw FixtureLoadError.missingResource
        }

        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw FixtureLoadError.decodeFailed
        }

        let decoded: FixtureBundlePayload
        do {
            decoded = try JSONDecoder().decode(FixtureBundlePayload.self, from: data)
        } catch {
            throw FixtureLoadError.decodeFailed
        }

        let source = inputPath == .tensor ? decoded.tensor : decoded.video
        guard let entry = source[clip.fixtureKey] else {
            throw FixtureLoadError.missingClipEntry
        }
        return entry
    }
}
