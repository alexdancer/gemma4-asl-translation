import Foundation

struct InferenceResult {
    let gloss: String
    let confidence: Double
}

protocol LocalCactusInferenceProviding {
    func infer() async throws -> InferenceResult
}

enum LocalInferenceError: Error {
    case missingFixture
    case invalidFixture
}

final class LocalCactusInferenceClient: LocalCactusInferenceProviding {
    func infer() async throws -> InferenceResult {
        // TODO: Replace fixture-backed response with real cactusComplete call
        guard let url = Bundle.main.url(forResource: "local_cactus_response", withExtension: "json") else {
            throw LocalInferenceError.missingFixture
        }

        let data = try Data(contentsOf: url)
        guard
            let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let gloss = payload["gloss"] as? String,
            let confidence = payload["confidence"] as? Double
        else {
            throw LocalInferenceError.invalidFixture
        }

        return InferenceResult(gloss: gloss, confidence: confidence)
    }
}
