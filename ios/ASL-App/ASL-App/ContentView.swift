import SwiftUI

@MainActor
final class InferenceViewModel: ObservableObject {
    @Published var glossText: String = "—"
    @Published var confidenceText: String = "—"
    @Published var statusText: String = "Tap to run local inference"

    private let client: LocalCactusInferenceProviding

    init(client: LocalCactusInferenceProviding) {
        self.client = client
    }

    func runInference() {
        Task {
            do {
                let result = try await client.infer()
                glossText = result.gloss
                confidenceText = String(format: "%.1f%%", result.confidence * 100)
                statusText = "Local Cactus inference complete"
            } catch {
                statusText = "Inference failed: \(error.localizedDescription)"
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel: InferenceViewModel

    init(viewModel: InferenceViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("ASL Tracer Slice #1")
                .font(.title2)

            Button("Run Local Cactus Inference") {
                viewModel.runInference()
            }
            .buttonStyle(.borderedProminent)

            Group {
                Text("Predicted Gloss: \(viewModel.glossText)")
                Text("Confidence: \(viewModel.confidenceText)")
                Text(viewModel.statusText)
                    .foregroundStyle(.secondary)
            }
            .font(.headline)

            Spacer()
        }
        .padding(20)
    }
}

#Preview {
    ContentView(viewModel: InferenceViewModel(client: LocalCactusInferenceClient()))
}
