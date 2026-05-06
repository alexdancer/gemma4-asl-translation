import Foundation

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

final class LocalCactusInferenceClient: LocalCactusInferenceProviding {
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
            // Try local runtime first, retry once, then fallback (unless strict proof mode is enabled).
            for attempt in 1...2 {
                if let local = await tryLocalRuntime(clip: clip, inputPath: inputPath, attempt: attempt, start: start) {
                    return local
                }
            }

            if strictProofMode {
                let latency = max(Int(Date().timeIntervalSince(start) * 1000), 1)
                return InferenceResult(
                    clipID: clip.rawValue,
                    inputPath: inputPath,
                    gloss: "",
                    confidence: 0,
                    latencyMs: latency,
                    runtimeMode: "local_cactus",
                    routeReason: "strict_proof_local_runtime_failed",
                    statusMessage: "Strict proof mode: local runtime failed (no fallback)",
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

    private func tryLocalRuntime(clip: DemoClip, inputPath: InputPath, attempt: Int, start: Date) async -> InferenceResult? {
        // TODO: Replace fixture-backed response with real cactusComplete call
        // TODO(issue-35): Replace this seam with real local cactusComplete runtime invocation.
        // Returning nil here simulates local runtime failure and exercises retry/fallback behavior.
        _ = (clip, inputPath, attempt)
        let _ = start
        return nil
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
