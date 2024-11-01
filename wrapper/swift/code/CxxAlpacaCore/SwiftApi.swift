// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
import CxxAlpacaCore

public enum ACError: Error, Equatable {
    case invalidModelCreation(String)
    case invalidInstanceCreation(String)
    case invalidRunOp(String)
}

/// Initializes the AlpacaCore native SDK
public func initSDK() {
    AC.initSDK()
}

class ProgressCallbackWrapper {
    let cb: Optional<(String, Float) -> Void>
    init(_ cb: Optional<(String, Float) -> Void>) {
        self.cb = cb
    }

    public func getRawPointer() -> UnsafeMutableRawPointer {
        return UnsafeMutableRawPointer(Unmanaged.passRetained(self).toOpaque())
    }

    public func releaseRawPointer(_ pointer: Optional<UnsafeMutableRawPointer>) {
        if pointer != nil {
            let unmanaged = Unmanaged<ProgressCallbackWrapper>.fromOpaque(pointer!)
            unmanaged.release()
        }
    }
}

func callObserver(observer: UnsafeMutableRawPointer,
                  tagBegin: UnsafePointer<Int8>,
                  tagEnd: UnsafePointer<Int8>,
                  progress: Float) {
    let wrapper = Unmanaged<ProgressCallbackWrapper>.fromOpaque(observer).takeUnretainedValue()
    if wrapper.cb != nil {
        let length = Int(tagEnd - tagBegin)
        let tagStr = String(data: Data(bytes: tagBegin, count: length), encoding: .utf8) ?? ""
        wrapper.cb!(tagStr, progress)
    }
}

/// Creates a model with the specified description and parameters, with optional progress tracking.
///
/// - Parameters:
///   - desc: A model description.
///   - params: A dictionary of parameters.
///   - progress: An optional closure for progress updates during model creation.
///               The closure parameters are:
///               - `String`: A description of the current progress step.
///               - `Float`: A value from 0.0 to 1.0 indicating the completion percentage.
///
/// - Throws: `ACError.invalidModelCreation` if the model creation process encounters an error.
///
/// - Returns: A `Model` instance representing the newly created model.
public func createModel(_ desc: inout ModelDesc, _ params: Dictionary<String, Any>,
                        _ progress: Optional<(String, Float) -> Void> = nil) throws -> Model {
    let paramsAsDict = try translateDictionaryToDict(params)
    let wrapper = ProgressCallbackWrapper(progress)
    var cbData: AC.ProgressCallbackData = AC.ProgressCallbackData()
    if progress != nil {
        cbData = AC.ProgressCallbackData(m_cb: callObserver, m_context: wrapper.getRawPointer())
    }

    var result = AC.createModel(&desc, paramsAsDict.getRef(), cbData)
    wrapper.releaseRawPointer(cbData.m_context)
    if result.hasError() {
        throw ACError.invalidModelCreation(String(result.error()))
    }
    return Model(result.consumeValue())
}

/// A wrapper class for the native `AC.Model` object, providing functionality to create new `Instance`.
public class Model {
    var model: AC.Model

    init(_ model: AC.Model) {
        self.model = model
    }

    /// Creates an `Instance` from the specified name and parameters.
    ///
    /// - Parameters:
    ///   - type: The type of the instance to create.
    ///   - params: A dictionary of parameters to pass for the instance creation.
    ///
    /// - Throws: `ACError.invalidInstanceCreation` if the instance creation fails.
    ///
    /// - Returns: An `Instance` representing the newly created instance.
    public func createInstance(_ type: String, _ params: Dictionary<String, Any>) throws -> Instance {
        let paramsAsDict = try translateDictionaryToDict(params)
        var result = model.createInstance(std.string(type), paramsAsDict.getRef())
        if result.hasError() {
            throw ACError.invalidInstanceCreation(String(result.error()))
        }
        return Instance(result.consumeValue())
    }
}

/// A wrapper class for native `AC.Instance` object, providing functionality to run operations with progress tracking.
public class Instance {
    var instance: AC.Instance

    init(_ instance: AC.Instance) {
        self.instance = instance
    }

    /// Runs an operation on the `instance` with the provided operation name, parameters, and optional progress callback.
    ///
    /// - Parameters:
    ///   - op: The name of the operation to execute.
    ///   - params: A dictionary of parameters to pass to the operation.
    ///   - progress: An optional closure that is called with updates on the operation's progress.
    ///               The closure parameters are:
    ///               - `String`: A description of the current progress step.
    ///               - `Float`: A value from 0.0 to 1.0 indicating the completion percentage.
    ///
    /// - Throws: An `ACError.invalidRunOp` error if the operation fails.
    ///
    /// - Returns: A dictionary containing the results of the operation.
    public func runOp(_ op: String, _ params: Dictionary<String, Any>,
                      _ progress: Optional<(String, Float) -> Void> = nil) throws -> Dictionary<String, Any> {
        let paramsAsDict = try translateDictionaryToDict(params)

        let wrapper = ProgressCallbackWrapper(progress)
        var cbData: AC.ProgressCallbackData = AC.ProgressCallbackData()
        if progress != nil {
            cbData = AC.ProgressCallbackData(m_cb: callObserver, m_context: wrapper.getRawPointer())
        }

        var result = instance.runOp(std.string(op), paramsAsDict.getRef(), cbData)
        wrapper.releaseRawPointer(cbData.m_context)

        if result.hasError() {
            throw ACError.invalidRunOp(String(result.error()))
        }
        return try translateDictToDictionary(result.consumeValue().getRef())
    }
}
