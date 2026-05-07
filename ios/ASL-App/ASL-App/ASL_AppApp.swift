import SwiftUI

@main
struct ASL_AppApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: InferenceViewModel(client: LocalCactusInferenceClient()))
        }
    }
}
