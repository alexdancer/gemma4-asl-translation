import SwiftUI

@main
struct ASLTracerSliceApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: InferenceViewModel(client: LocalCactusInferenceClient()))
        }
    }
}
