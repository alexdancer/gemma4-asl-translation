import Foundation

#if canImport(cactus)
import cactus

typealias CactusModelT = cactus_model_t

enum CactusShimError: Error {
    case initFailed
    case completeFailed(Int32, String)
}

@inline(__always)
func cactusInit(_ modelPath: String, _ corpusDir: String?, _ cacheIndex: Bool) throws -> CactusModelT {
    guard let model = cactus_init(modelPath, corpusDir, cacheIndex) else {
        throw CactusShimError.initFailed
    }
    return model
}

@inline(__always)
func cactusDestroy(_ model: CactusModelT) {
    cactus_destroy(model)
}

@inline(__always)
func cactusComplete(
    _ model: CactusModelT,
    _ messagesJSON: String,
    _ optionsJSON: String?,
    _ toolsJSON: String?,
    _ callback: cactus_token_callback?,
    _ userData: UnsafeMutableRawPointer?
) throws -> String {
    let bufferSize = 64 * 1024
    var responseBuffer = Array<CChar>(repeating: 0, count: bufferSize)

    let rc = cactus_complete(
        model,
        messagesJSON,
        &responseBuffer,
        responseBuffer.count,
        optionsJSON,
        toolsJSON,
        callback,
        userData,
        nil,
        0
    )

    guard rc == 0 else {
        let errPtr = cactus_get_last_error()
        let err = errPtr.map { String(cString: $0) } ?? "unknown cactus error"
        throw CactusShimError.completeFailed(rc, err)
    }

    return String(cString: responseBuffer)
}
#endif

struct InferenceResult {
    let clipID: String
    let inputPath: InputPath
    let gloss: String
    let translation: String
    let confidence: Double
    let latencyMs: Int
    let runtimeMode: String
    let routeReason: String
    let requestID: String
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
    case cloud = "Cloud"

    var id: String { rawValue }
}

protocol LocalCactusInferenceProviding {
    func infer(
        clip: DemoClip,
        inputPath: InputPath,
        runtimeMode: RuntimeMode,
        strictProofMode: Bool,
        uploadVideoData: Data?,
        uploadFilename: String?
    ) async -> InferenceResult
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
    case modelInitFailed(String)
    case responseDecodeFailed
    case invocationFailed(String)
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
    #if canImport(cactus)
    private var model: CactusModelT?
    #endif

    func debugResolvedModelPath() -> String? {
        #if canImport(cactus)
        return resolveModelPath()
        #else
        return nil
        #endif
    }

    func infer(clip: DemoClip, inputPath: InputPath) async throws -> LocalCactusPrediction {
        #if canImport(cactus)
        let modelHandle = try loadModelIfNeeded()
        let messages = try makeMessagesJSON(clip: clip, inputPath: inputPath)
        let options = "{\"max_tokens\":64,\"temperature\":0.0}"

        let responseJson: String
        do {
            responseJson = try cactusComplete(modelHandle, messages, options, nil, nil, nil)
        } catch let error as CactusShimError {
            switch error {
            case .completeFailed(let code, let message):
                throw CactusRuntimeError.invocationFailed("cactus_complete rc=\(code): \(message)")
            case .initFailed:
                throw CactusRuntimeError.invocationFailed("unexpected init failure inside complete path")
            }
        } catch {
            throw CactusRuntimeError.invocationFailed("unknown completion error: \(error.localizedDescription)")
        }

        guard let data = responseJson.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(CactusCompletionResponse.self, from: data)
        else {
            throw CactusRuntimeError.responseDecodeFailed
        }

        guard decoded.success else {
            throw CactusRuntimeError.invocationFailed("runtime returned success=false; payload=\(responseJson.prefix(200))")
        }

        let raw = (decoded.response ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard raw.isEmpty == false else {
            throw CactusRuntimeError.invocationFailed("runtime returned empty response; payload=\(responseJson.prefix(200))")
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

    #if canImport(cactus)
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
            let errPtr = cactus_get_last_error()
            let err = errPtr.map { String(cString: $0) } ?? "unknown cactus_init error"
            throw CactusRuntimeError.modelInitFailed(err)
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
        #if canImport(cactus)
        return runtimeAdapter.debugResolvedModelPath() ?? "unresolved"
        #else
        return "sdk_unavailable"
        #endif
    }

    func infer(
        clip: DemoClip,
        inputPath: InputPath,
        runtimeMode: RuntimeMode,
        strictProofMode: Bool,
        uploadVideoData: Data?,
        uploadFilename: String?
    ) async -> InferenceResult {
        _ = strictProofMode
        _ = runtimeMode
        let started = Date()
        let requestID = UUID().uuidString
        let maxRetryAttempts = 1

        for attempt in 0...maxRetryAttempts {
            do {
                guard let endpoint = resolveCloudEndpoint() else {
                    return failedResult(
                        clip: clip,
                        inputPath: inputPath,
                        started: started,
                        requestID: requestID,
                        routeReason: "cloud_endpoint_not_configured",
                        statusMessage: "Cloud endpoint is not configured. Set ASL_CLOUD_ENDPOINT (or Info.plist ASLCloudEndpoint) to your deployed /v1/translate-sign URL.",
                        expectedGloss: (try? expectedGlossFor(clip: clip, inputPath: inputPath)) ?? ""
                    )
                }

                if isPhysicalDeviceLoopback(endpoint: endpoint) {
                    return failedResult(
                        clip: clip,
                        inputPath: inputPath,
                        started: started,
                        requestID: requestID,
                        routeReason: "cloud_endpoint_loopback_on_device",
                        statusMessage: "Endpoint points to localhost on iPhone. Use a reachable cloud URL for /v1/translate-sign.",
                        expectedGloss: (try? expectedGlossFor(clip: clip, inputPath: inputPath)) ?? ""
                    )
                }

                var request = URLRequest(url: endpoint)
                request.httpMethod = "POST"
                request.timeoutInterval = 12
                request.setValue(requestID, forHTTPHeaderField: "X-Request-ID")

                let boundary = "Boundary-\(UUID().uuidString)"
                request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
                request.httpBody = makeMultipartBody(
                    boundary: boundary,
                    clip: clip,
                    uploadVideoData: uploadVideoData,
                    uploadFilename: uploadFilename
                )

                let (data, response) = try await URLSession.shared.data(for: request)
                let http = response as? HTTPURLResponse

                if let http, (200..<300).contains(http.statusCode) {
                    let decoded = try JSONDecoder().decode(CloudSuccessResponse.self, from: data)
                    return InferenceResult(
                        clipID: clip.rawValue,
                        inputPath: inputPath,
                        gloss: decoded.gloss,
                        translation: decoded.translation,
                        confidence: decoded.confidence,
                        latencyMs: decoded.latencyMs,
                        runtimeMode: "cloud",
                        routeReason: "cloud_endpoint_success",
                        requestID: decoded.requestId ?? requestID,
                        statusMessage: "Cloud translation complete",
                        expectedGloss: (try? expectedGlossFor(clip: clip, inputPath: inputPath)) ?? "",
                        success: true
                    )
                }

                let failure = (try? JSONDecoder().decode(CloudErrorResponse.self, from: data))
                let canRetry = (failure?.retryable ?? false) && attempt < maxRetryAttempts
                if canRetry {
                    continue
                }

                let latency = max(Int(Date().timeIntervalSince(started) * 1000), 1)
                return InferenceResult(
                    clipID: clip.rawValue,
                    inputPath: inputPath,
                    gloss: "",
                    translation: "",
                    confidence: 0,
                    latencyMs: latency,
                    runtimeMode: "cloud",
                    routeReason: attempt > 0 ? "cloud_endpoint_retry_exhausted" : "cloud_endpoint_error",
                    requestID: failure?.requestId ?? requestID,
                    statusMessage: mapCloudErrorMessage(errorCode: failure?.errorCode, fallbackMessage: failure?.message, exhaustedRetry: attempt >= maxRetryAttempts),
                    expectedGloss: (try? expectedGlossFor(clip: clip, inputPath: inputPath)) ?? "",
                    success: false
                )
            } catch {
                if attempt < maxRetryAttempts {
                    continue
                }

                let latency = max(Int(Date().timeIntervalSince(started) * 1000), 1)
                return InferenceResult(
                    clipID: clip.rawValue,
                    inputPath: inputPath,
                    gloss: "",
                    translation: "",
                    confidence: 0,
                    latencyMs: latency,
                    runtimeMode: "cloud",
                    routeReason: "cloud_endpoint_unreachable",
                    requestID: requestID,
                    statusMessage: "Could not connect to server. Please check network and try again.",
                    expectedGloss: (try? expectedGlossFor(clip: clip, inputPath: inputPath)) ?? "",
                    success: false
                )
            }
        }

        let latency = max(Int(Date().timeIntervalSince(started) * 1000), 1)
        return InferenceResult(
            clipID: clip.rawValue,
            inputPath: inputPath,
            gloss: "",
            translation: "",
            confidence: 0,
            latencyMs: latency,
            runtimeMode: "cloud",
            routeReason: "cloud_endpoint_retry_exhausted",
            requestID: requestID,
            statusMessage: "Request failed after retry. Please try again.",
            expectedGloss: (try? expectedGlossFor(clip: clip, inputPath: inputPath)) ?? "",
            success: false
        )
    }

    private func resolveCloudEndpoint() -> URL? {
        if let env = ProcessInfo.processInfo.environment["ASL_CLOUD_ENDPOINT"], env.isEmpty == false {
            return URL(string: env)
        }
        if let plist = Bundle.main.object(forInfoDictionaryKey: "ASLCloudEndpoint") as? String, plist.isEmpty == false {
            return URL(string: plist)
        }
        return nil
    }

    private func isPhysicalDeviceLoopback(endpoint: URL) -> Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        guard let host = endpoint.host?.lowercased() else { return false }
        return host == "127.0.0.1" || host == "localhost"
        #endif
    }

    private func failedResult(
        clip: DemoClip,
        inputPath: InputPath,
        started: Date,
        requestID: String,
        routeReason: String,
        statusMessage: String,
        expectedGloss: String
    ) -> InferenceResult {
        let latency = max(Int(Date().timeIntervalSince(started) * 1000), 1)
        return InferenceResult(
            clipID: clip.rawValue,
            inputPath: inputPath,
            gloss: "",
            translation: "",
            confidence: 0,
            latencyMs: latency,
            runtimeMode: "cloud",
            routeReason: routeReason,
            requestID: requestID,
            statusMessage: statusMessage,
            expectedGloss: expectedGloss,
            success: false
        )
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

    private func makeMultipartBody(
        boundary: String,
        clip: DemoClip,
        uploadVideoData: Data?,
        uploadFilename: String?
    ) -> Data {
        var data = Data()
        let payload = uploadVideoData ?? ("demo-video-bytes-\(clip.fixtureKey)".data(using: .utf8) ?? Data())
        let filename = uploadFilename ?? "\(clip.fixtureKey).mov"
        let ext = URL(fileURLWithPath: filename).pathExtension.lowercased()
        let mimeType = ext == "mp4" ? "video/mp4" : "video/quicktime"

        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"video\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        data.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
        data.append(payload)
        data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        return data
    }

    private func mapCloudErrorMessage(errorCode: String?, fallbackMessage: String?, exhaustedRetry: Bool) -> String {
        switch errorCode {
        case "CLIP_TOO_LONG":
            return "Clip is longer than 5 seconds. Please choose a shorter clip."
        case "UNSUPPORTED_FORMAT":
            return "Unsupported video format. Please use .mov or .mp4."
        case "RATE_LIMITED":
            return exhaustedRetry ? "Service is busy right now. Please try again shortly." : "Service is busy. Retrying once automatically..."
        case "TIMEOUT":
            return "Translation timed out. Please try again."
        default:
            return fallbackMessage ?? "Cloud request failed"
        }
    }
}

private struct CloudSuccessResponse: Decodable {
    let requestId: String?
    let gloss: String
    let translation: String
    let confidence: Double
    let latencyMs: Int

    private enum CodingKeys: String, CodingKey {
        case requestId = "request_id"
        case gloss
        case translation
        case confidence
        case latencyMs = "latency_ms"
    }
}

private struct CloudErrorResponse: Decodable {
    let errorCode: String
    let message: String
    let requestId: String?
    let retryable: Bool

    private enum CodingKeys: String, CodingKey {
        case errorCode = "error_code"
        case message
        case requestId = "request_id"
        case retryable
    }
}
