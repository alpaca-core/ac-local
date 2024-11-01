/// Represents the information of an asset.
public class AssetInfo {
    /// The path to the asset.
    public var path: String = ""

    /// The tag of the asset. Some native model loaders may require it.
    public var tag: String = ""

    public init() {}

    public init(_ path:String, _ tag:String) {
        self.path = path
        self.tag = tag
    }
}

/// The description of the model used by the native `ModelLoader`.
public class ModelDesc {
    /// The type of the inference used to select the appropriate model loader.
    public var inferenceType: String = ""

    /// Name tag. Not used by the library in any way besides logs and may be helpful for debugging
    public var name: String = ""

    /// The list of assets required by the model.
    public var assets: [AssetInfo]

    public init() {
        self.assets = []
    }

    public init(_ inferenceType:String = "", _ name:String = "", _ assets:[AssetInfo] = []) {
        self.inferenceType = inferenceType
        self.name = name
        self.assets = assets
    }
}
